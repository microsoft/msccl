/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nvmlwrap.h"
#include "net.h"
#include "coll_net.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "xml.h"
#include "cpuset.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };
const char* topoLinkTypeStr[] = { "LOC", "NVL", "",    "PCI", "",    "",    "SYS", "NET" };
const char* topoPathTypeStr[] = { "LOC", "NVL", "NVB", "PIX", "PXB", "PHB", "SYS" };

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  char* str = path+offset;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  int64_t numid;
  NCCLCHECK(busIdToInt64(str, &numid));
  // Ignore subdevice because those should use the same PCI link so we want to merge nodes.
  numid -= numid & 0xf;
  *id = numid;
  return ncclSuccess;
}

static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu) {
  *cpu = NULL;
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  for (int l=0; l<node->nlinks; l++) {
    if (node->links[l].type == LINK_PCI) NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu));
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuWidth = 0;
int cpuPciWidth = 0;

static ncclResult_t ncclTopoGetInterCpuWidth(struct ncclTopoNode* cpu, float* width) {
  *width = LOC_WIDTH;
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *width = P9_WIDTH;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *width = ARM_WIDTH;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *width = cpu->cpu.model == NCCL_TOPO_CPU_TYPE_SKL ? SKL_QPI_WIDTH : QPI_WIDTH;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *width = cpu->cpu.model ==  NCCL_TOPO_CPU_TYPE_YONGFENG ? YONGFENG_ZPI_WIDTH : ZPI_WIDTH;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    // Create link to itself (used in some corner cases)
    n->nlinks=1;
    n->links[0].type = LINK_LOC;
    n->links[0].remNode = n;
    n->links[0].width = LOC_WIDTH;
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.width = 0.0;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    free(delNode->paths[t]);
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      if (node == delNode) continue;
      for (int l=0; l<node->nlinks; l++) {
        while (l<node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (l<node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, float width) {
  // Aggregate links into higher width for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->width += width;

  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link-1)->width >= linkSave.width) break;
    memcpy(link, link-1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    for (int p=0; p<system->nodes[CPU].count; p++) {
      if (n == p) continue;
      float width;
      NCCLCHECK(ncclTopoGetInterCpuWidth(system->nodes[CPU].nodes+n, &width));
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, system->nodes[CPU].nodes+p, LINK_SYS, width));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%lX (%d)", topoNodeTypeStr[node->type], node->id, node->gpu.rank);
  } else if (node->type == CPU) {
    sprintf(line+offset, "%s/%lX (%d/%d/%d)", topoNodeTypeStr[node->type], node->id, node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else {
    sprintf(line+offset, "%s/%lX", topoNodeTypeStr[node->type], node->id);
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) continue;
    if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->width);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line+nextOffset, "%s/%lX (%lx/%d/%f)", topoNodeTypeStr[link->remNode->type], link->remNode->id, link->remNode->net.asic, link->remNode->net.port, link->remNode->net.width);
        } else {
          sprintf(line+nextOffset, "%s/%lX", topoNodeTypeStr[link->remNode->type], link->remNode->id);
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxWidth %2.1f totalWidth %2.1f ===", s->maxWidth, s->totalWidth);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. SYS (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic) {
  int dev;
  NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &dev));

  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, dev));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  if (str) sscanf(str, "0x%lx", &net->net.asic);
  else net->net.asic = dev;

  ncclDebugNoWarn = NCCL_GRAPH;
  int mbps;
  if (xmlGetAttrInt(xmlNet, "speed", &mbps) != ncclSuccess) mbps = 0;
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.width = mbps / 8000.0;
  if (xmlGetAttrInt(xmlNet, "port", &net->net.port) != ncclSuccess) net->net.port = 0;
  if (xmlGetAttrInt(xmlNet, "gdr", &net->net.gdrSupport) != ncclSuccess) net->net.gdrSupport = 0;
  if (xmlGetAttrInt(xmlNet, "maxconn", &net->net.maxChannels) != ncclSuccess) net->net.maxChannels = MAXCHANNELS;
  if (xmlGetAttrInt(xmlNet, "coll", &net->net.collSupport) != ncclSuccess) net->net.collSupport = 0;
  ncclDebugNoWarn = 0;

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, net->net.width));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, net->net.width));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic) {
  for (int s=0; s<xmlNic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    if (index == -1) continue;
    NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.cudaCompCap));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  // Do not go any further, nvlinks will be added in a second pass
  return ncclSuccess;
}

struct kvDict kvDictPciClass[] = { { "0x060400", PCI }, { "0x068000", NVS }, { "0x068001", CPU }, { "0x03", GPU }, { "0x02", NIC }, { NULL, PCI /* Default fallback value */ } };
struct kvDict kvDictPciGen[] = { { "2.5 GT/s", 15 }, { "5 GT/s", 30 }, { "8 GT/s", 60 }, { "16 GT/s", 120 }, { NULL, 60 /* Default fallback */ } }; // x100 Mbps per lane
ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent) {
  const char* str;

  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlGpu = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
  if (xmlGpu != NULL) {
    type = GPU;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    if (index == -1) return ncclSuccess;
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, busId));
    NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, node));
  }
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    busId &= 0xfffffffffffffff0;
    struct ncclTopoNode* nicNode = NULL;
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, busId));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, busId));
      node = nicNode; // Connect it to parent later on
    }
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode));
  } else if (type == PCI) {
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, busId));
    for (int s=0; s<xmlPci->nSubs; s++) {
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node));
    }
  }

  if (node) {
    int width, speed;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));

    // Manage cases where speed was not indicated in /sys
    if (width == 0) width = 16;
    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)

    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, width*speed/80.0));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, width*speed/80.0));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuArch[] = { { "x86_64", NCCL_TOPO_CPU_ARCH_X86 }, { "arm64", NCCL_TOPO_CPU_ARCH_ARM }, { "ppc64", NCCL_TOPO_CPU_ARCH_POWER }, { NULL, 0 } };
struct kvDict kvDictCpuVendor[] = { { "GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL }, { "AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD }, { "CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { "  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { NULL, 0 } };

ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int numaId;
  NCCLCHECK(xmlGetAttrInt(xmlCpu, "numaid", &numaId));
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU, numaId));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.model = (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_TYPE_SKL : NCCL_TOPO_CPU_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_TYPE_YONGFENG;
    }
  }
  for (int s=0; s<xmlCpu->nSubs; s++) {
    struct ncclXmlNode* node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu));
    if (strcmp(node->name, "nic") == 0) {
      struct ncclTopoNode* nic = NULL;
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, 0));
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, 0));
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, LOC_WIDTH));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, LOC_WIDTH));
      }
      NCCLCHECK(ncclTopoAddNic(node, system, nic));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId) {
  if (strcmp(node->name, "nvlink") == 0) {
    struct ncclTopoNode* gpu = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      WARN("Add NVLink error : could not find GPU %lx", pBusId);
      return ncclInternalError;
    }
    int count;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    const char* targetClass;
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    int targetType;
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    struct ncclTopoNode* remote = NULL;
    if (targetType == GPU) {
      // NVL P2P connection to another GPU
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      NCCLCHECK(ncclTopoGetNode(system, &remote, GPU, busId));
    } else if (targetType == CPU) {
      // NVL connection to the local CPU
      NCCLCHECK(findLocalCpu(gpu, &remote));
    } else {
      if (system->nodes[NVS].count == 0) {
        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
      } else {
        remote = system->nodes[NVS].nodes;
      }
    }
    if (remote) {
      float nvlSpeed = ncclTopoNVLinkSpeed(gpu->gpu.cudaCompCap);
      NCCLCHECK(ncclTopoConnectNodes(gpu, remote, LINK_NVL, count*nvlSpeed));
      if (remote->type != GPU) {
        NCCLCHECK(ncclTopoConnectNodes(remote, gpu, LINK_NVL, count*nvlSpeed));
      }
    }
  } else {
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }
  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL));

  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

// Only set values if not already set
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}


ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system) {
  struct ncclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  char* xmlTopoFile = getenv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml));
  }
  if (xml->maxIndex == 0) {
    // Create top tag
    struct ncclXmlNode* top;
    NCCLCHECK(xmlAddNode(xml, NULL, "system", &top));
    NCCLCHECK(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION));
  }

  // Auto-detect GPUs if needed
  for (int r=0; r<comm->nRanks; r++) {
    if (comm->peerInfo[r].hostHash == comm->peerInfo[comm->rank].hostHash) {
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      NCCLCHECK(int64ToBusId(comm->peerInfo[r].busId, busId));
      struct ncclXmlNode* node;
      NCCLCHECK(ncclTopoFillGpu(xml, busId, &node));
      if (node == NULL) continue;
      NCCLCHECK(xmlSetAttrInt(node, "keep", 1));
      NCCLCHECK(xmlSetAttrInt(node, "rank", r));
      NCCLCHECK(xmlInitAttrInt(node, "gdr", comm->peerInfo[r].gdrSupport));
    }
  }
  // Auto-detect NICs if needed. net/collnet share the same xml/graph nodes,
  // so we start with collnet so that it has precedence.
  int netDevCount = 0;
  if (ncclCollNet) {
    NCCLCHECK(collNetDevices(&netDevCount));
    for (int n=0; n<netDevCount; n++) {
      ncclNetProperties_t props;
      NCCLCHECK(collNetGetProperties(n, &props));
      struct ncclXmlNode* netNode;
      NCCLCHECK(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode));
      NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
      NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
      NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
      NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
      NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
      NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
      NCCLCHECK(xmlInitAttrInt(netNode, "gdr", props.ptrSupport & NCCL_PTR_CUDA ? 1 : 0));
      NCCLCHECK(xmlInitAttrInt(netNode, "coll", 1));
    }
  }
  if (netDevCount == 0) {
    NCCLCHECK(ncclNetDevices(&netDevCount));
  }
  for (int n=0; n<netDevCount; n++) {
    ncclNetProperties_t props;
    NCCLCHECK(ncclNetGetProperties(n, &props));
    struct ncclXmlNode* netNode;
    NCCLCHECK(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode));
    NCCLCHECK(xmlSetAttrInt(netNode, "keep", 1));
    NCCLCHECK(xmlSetAttrInt(netNode, "dev", n));
    NCCLCHECK(xmlInitAttrInt(netNode, "speed", props.speed));
    NCCLCHECK(xmlInitAttrInt(netNode, "port", props.port));
    NCCLCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    NCCLCHECK(xmlInitAttrInt(netNode, "maxconn", props.maxComms));
    NCCLCHECK(xmlInitAttrInt(netNode, "gdr", props.ptrSupport & NCCL_PTR_CUDA ? 1 : 0));
  }

  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  NCCLCHECK(ncclTopoTrimXml(xml));

  xmlTopoFile = getenv("NCCL_TOPO_DUMP_FILE");
  if (xmlTopoFile && comm->rank == ncclParamTopoDumpFileRank()) {
    INFO(NCCL_ENV, "NCCL_TOPO_DUMP_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECK(ncclTopoDumpXmlToFile(xmlTopoFile, xml));
  }

  NCCLCHECK(ncclTopoGetSystemFromXml(xml, system));
  free(xml);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int64_t* id, int rr) {
  int g;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &g));
  int minType = PATH_SYS;
  float maxWidth = 0;
  int count = 0;
  int* nets;
  NCCLCHECK(ncclCalloc(&nets, system->nodes[NET].count));
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = system->nodes[NET].nodes[n].paths[GPU]+g;
    if (path->width > maxWidth || (path->width == maxWidth && path->type < minType)) {
      maxWidth = path->width;
      minType = path->type;
      count = 0;
    }
    if (path->width == maxWidth && path->type == minType) nets[count++] = system->nodes[NET].nodes[n].id;
  }
  *id = nets[rr % count];

  free(nets);
  return ncclSuccess;
}

ncclResult_t scclGetBufferType(const char* str, uint8_t* output){
  if (strcmp(str, "i") == 0){
    *output = SCCL_INPUT_BUFFER;
  } else if (strcmp(str, "o") == 0) {
    *output = SCCL_OUTPUT_BUFFER;
  } else if (strcmp(str, "s") == 0) {
    *output = SCCL_SCRATCH_BUFFER;
  } else {
    WARN("type of buffer is not supported: %s", str);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t scclCheckBufferBounds(int bufferType, int offset, int nInputChunks, int nOutputChunks, int nScratchChunks){
  if (bufferType == SCCL_INPUT_BUFFER){
    if (offset < -1 || offset >= nInputChunks){
      WARN("Incorrect offset set for input buffer: offset: %d maximum allowed: %d", offset, nInputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == SCCL_OUTPUT_BUFFER){
    if (offset < -1 || offset >= nOutputChunks){
      WARN("Incorrect offset set for output buffer: offset: %d maximum allowed: %d", offset, nOutputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == SCCL_SCRATCH_BUFFER){
    if (offset < -1 || offset >= nScratchChunks){
      WARN("Incorrect offset set for scratch buffer: offset: %d maximum allowed: %d", offset, nScratchChunks);
      return ncclInvalidUsage;
    }
  }
  return ncclSuccess;
}

ncclResult_t scclProtocolStrToId(const char *protocol, int *protocolId) {
  if (strcmp(protocol, "Simple") == 0){
    *protocolId = NCCL_PROTO_SIMPLE;
  } else if (strcmp(protocol, "LL128") == 0){
    *protocolId = NCCL_PROTO_LL128;
  } else if (strcmp(protocol, "LL") == 0){
    *protocolId = NCCL_PROTO_LL;
  } else {
    WARN("SCCL: protocol %s is not supported.", protocol);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t scclGetAlgoFromXMLAndSetComm(struct ncclComm* comm, const char* str, struct scclAlgorithm* scclAlgo) {
  INFO(NCCL_INIT, "SCCL: Parsing algorithm %s", str);
  struct ncclXml* xml;

  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(scclGetXmlAlgoFromFile(str, xml));
  int rank = comm->rank;

  // zeroing out all entries.
  memset(scclAlgo, 0, sizeof(struct scclAlgorithm));
  scclAlgo->isValid = false; // set isValid to false until we hit the return ncclSuccess.
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "algo", &topNode));
  const char* name;
  NCCLCHECK(xmlGetAttrStr(topNode, "name", &name));
  strncpy(scclAlgo->name, name, SCCL_MAX_ALGO_NAME);

  int ngpus;
  NCCLCHECK(xmlGetAttrInt(topNode, "ngpus", &ngpus));
  if (comm->nRanks != ngpus){
    WARN("SCCL: ngpus set in the SCCL algo (%d) doesn't match the communicator ngpus (%d)", ngpus, comm->nRanks);
    return ncclInvalidUsage;
  }
  scclAlgo->ngpus = ngpus;
  int nchunksPerLoop;
  NCCLCHECK(xmlGetAttrInt(topNode, "nchunksperloop", &nchunksPerLoop));
  int globalNChannels;
  NCCLCHECK(xmlGetAttrInt(topNode, "nchannels", &globalNChannels));

  int redopExists = 0;
  NCCLCHECK(xmlAttrExists(topNode, "redop", &redopExists));
  if (redopExists){
    const char* redop;
    // redop exists
    NCCLCHECK(xmlGetAttrStr(topNode, "redop", &redop));
    if (strcmp(redop, "sum") == 0){
      scclAlgo->redOp = ncclSum;
    } else if (strcmp(redop, "prod") == 0){
      scclAlgo->redOp = ncclProd;
    } else if (strcmp(redop, "max") == 0){
      scclAlgo->redOp = ncclMax;
    } else if (strcmp(redop, "min") == 0){
      scclAlgo->redOp = ncclMin;
    } else if (strcmp(redop, "nop") == 0){
      //If algorithm has no reduction operator then use ncclSum.
      scclAlgo->redOp = ncclSum;
    } else {
      WARN("SCCL: redop %s is not supported.", redop);
      return ncclInvalidUsage;
    }
  } else {
    // redop doesn't exist, default to nop/ncclSum
    scclAlgo->redOp = ncclSum;
  }

  const char* protocol;
  NCCLCHECK(xmlGetAttrStr(topNode, "proto", &protocol));
  NCCLCHECK(scclProtocolStrToId(protocol, &scclAlgo->protocol));

  int minBytesExists = 0;
  NCCLCHECK(xmlAttrExists(topNode, "minBytes", &minBytesExists));
  int64_t minBytes;
  if (minBytesExists) {
    NCCLCHECK(xmlGetAttrInt64_t(topNode, "minBytes", &minBytes));
  } else {
    minBytes = 0;
  }

  int maxBytesExists = 0;
  NCCLCHECK(xmlAttrExists(topNode, "maxBytes", &maxBytesExists));
  int64_t maxBytes;
  if (maxBytesExists) {
    NCCLCHECK(xmlGetAttrInt64_t(topNode, "maxBytes", &maxBytes));
  } else {
    maxBytes = (((int64_t)1)<<35); // set max to 32 GB which is sufficient for now.
  }
  if (minBytes > maxBytes) {
    WARN("SCCL: minBytes cannot be greater than maxBytes.");
    return ncclInvalidUsage;
  }
  if (minBytes < 0) {
    WARN("SCCL: minBytes cannot be negative.");
    return ncclInvalidUsage;
  }
  if (maxBytes < 0) {
    WARN("SCCL: maxBytes cannot be negative.");
    return ncclInvalidUsage;
  }
  scclAlgo->minBytes = minBytes;
  scclAlgo->maxBytes = maxBytes;

  const char* collectiveType;
  NCCLCHECK(xmlGetAttrStr(topNode, "coll", &collectiveType));
  if (strcmp(collectiveType, "allreduce") == 0){
    scclAlgo->collectiveType = ncclFuncAllReduce;
  } else if (strcmp(collectiveType, "allgather") == 0){
    scclAlgo->collectiveType = ncclFuncAllGather;
  } else if (strcmp(collectiveType, "reduce") == 0){
    scclAlgo->collectiveType = ncclFuncReduce;
  } else if (strcmp(collectiveType, "broadcast") == 0){
    scclAlgo->collectiveType = ncclFuncBroadcast;
  } else if (strcmp(collectiveType, "alltoall") == 0){
    scclAlgo->collectiveType = ncclFuncAllToAll;
  } else if (strcmp(collectiveType, "reduce_scatter") == 0){
    scclAlgo->collectiveType = ncclFuncReduceScatter;
  } else if (strcmp(collectiveType, "custom") == 0){
    scclAlgo->collectiveType = ncclFuncCustomCollective;
  } else {
    WARN("SCCL: collective type %s is not supported.", collectiveType);
    return ncclInvalidUsage;
  }

  int inplace;
  NCCLCHECK(xmlGetAttrInt(topNode, "inplace", &inplace));
  if (inplace) {
    scclAlgo->inPlace = 1;
  } else {
    scclAlgo->inPlace = 0;
  }

  scclAlgo->nChannels = globalNChannels;
  scclAlgo->nchunksPerLoop  = nchunksPerLoop;
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "gpu") == 0){
      int blockExists[SCCL_MAX_NUM_THREAD_BLOCKS];
      memset(blockExists, 0, sizeof(int[SCCL_MAX_NUM_THREAD_BLOCKS]));
      int id, nScratchChunks, nInputChunks, nOutputChunks;
      NCCLCHECK(xmlGetAttrInt(node, "id", &id));
      if (id == rank){
        NCCLCHECK(xmlGetAttrInt(node, "i_chunks", &nInputChunks));
        NCCLCHECK(xmlGetAttrInt(node, "o_chunks", &nOutputChunks));
        NCCLCHECK(xmlGetAttrInt(node, "s_chunks", &nScratchChunks));
        if (nScratchChunks < 0){
          WARN("SCCL: nScratchChunks must be not negative. nScratchChunks: %d", nScratchChunks);
          return ncclInvalidUsage;
        }
        scclAlgo->nScratchChunks = nScratchChunks;
        for (int t=0; t<node->nSubs; t++) {
          struct ncclXmlNode* threadblockNode = node->subs[t];
          if (strcmp(threadblockNode->name, "tb") == 0){
            int bid, recvpeer, sendpeer, channelId;
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "id", &bid));
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "recv", &recvpeer));
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "send", &sendpeer));
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "chan", &channelId));
            if (bid < 0){
              WARN("SCCL: bid must be not negative. bid: %d", bid);
              return ncclInvalidUsage;
            }              
            if (bid >= SCCL_MAX_NUM_THREAD_BLOCKS){
              WARN("SCCL: too many thread blocks are requested. Max thread blocks: %d", SCCL_MAX_NUM_THREAD_BLOCKS);
              return ncclInvalidUsage;
            }
            if (blockExists[bid]){
              WARN("SCCL: duplicate thread block id %d for SCCL", bid);
              return ncclInvalidUsage;
            }
            blockExists[bid] = 1;

            if (recvpeer == id || sendpeer == id){
              WARN("SCCL: peer (%d,%d) and gpu id (%d) must be different", recvpeer, sendpeer, id);
              return ncclInvalidUsage;
            }
            struct scclThreadBlock* sTB = &scclAlgo->scclTB[bid];
            sTB->nsteps = 0;
            if (recvpeer < -1 || sendpeer < -1){
              WARN("SCCL: wrong recvpeer (%d) or sendpeer (%d) in threadblock %d on gpu %d", recvpeer, sendpeer, bid, id);
              return ncclInvalidUsage;
            }

            if (recvpeer == id || sendpeer == id){
              WARN("SCCL: recvpeer (%d) or sendpeer (%d) for threadblock %d cannot be gpu %d", recvpeer, sendpeer, bid, id);
              return ncclInvalidUsage;
            }

            if (recvpeer >= ngpus || sendpeer >= ngpus) {
              WARN("SCCL: recvpeer (%d) or sendpeer (%d) must be -1 or between 0 and ngpus (%d)", recvpeer, sendpeer, ngpus);
              return ncclInvalidUsage;
            }

            sTB->recvpeer = recvpeer;
            sTB->sendpeer = sendpeer;
            if (channelId < 0 || channelId > MAXCHANNELS){
              if (channelId == -1 && recvpeer == -1 && sendpeer == -1){
                WARN("SCCL: threadblock %d on GPU %d has no send or recv", bid, id);
              } else {
                WARN("SCCL for threadblocks with recv/send, chan needs to be between 0 and %d and it was %d", MAXCHANNELS, channelId);
                return ncclInvalidUsage;
              }
            }
            sTB->channelId = channelId;

            // setting the summary of the sccl aglorithm in sccl channels
            scclChannelInfo* scclChannel = (sTB->channelId == -1) ? NULL : &scclAlgo->scclChannels[sTB->channelId];

            int numDependences = 0;
            int oldDependencePointer = 0; // inidcator of where the dependences started for nop

            int oldReductionDstBuffer = -1; // Indicator of last reduction buffer name; -1 means that last one wasn't a compatible reduction
            int oldReductionDstOffset = -1; // Indicator of last reduction buffer index
            int oldReductionSrcBuffer = -1; // 
            int numReductions = 0;

            int numTransfers = 0;
            for (int st=0; st<threadblockNode->nSubs; st++) {
              struct ncclXmlNode* stepNode = threadblockNode->subs[st];
              if (strcmp(stepNode->name, "step") == 0){
                int s, srcoffset, dstoffset, depend_bid, depend_step, has_dependence, count;
                const char* srcbuffer, * dstbuffer, * type;
                NCCLCHECK(xmlGetAttrInt(stepNode, "s", &s));

                NCCLCHECK(xmlGetAttrInt(stepNode, "srcoff", &srcoffset));
                NCCLCHECK(xmlGetAttrStr(stepNode, "srcbuf", &srcbuffer));
                NCCLCHECK(xmlGetAttrInt(stepNode, "dstoff", &dstoffset));
                NCCLCHECK(xmlGetAttrStr(stepNode, "dstbuf", &dstbuffer));

                NCCLCHECK(xmlGetAttrInt(stepNode, "cnt", &count));
                NCCLCHECK(xmlGetAttrStr(stepNode, "type", &type));
                NCCLCHECK(xmlGetAttrInt(stepNode, "depid", &depend_bid));
                NCCLCHECK(xmlGetAttrInt(stepNode, "deps", &depend_step));
                NCCLCHECK(xmlGetAttrInt(stepNode, "hasdep", &has_dependence));

                if (s >= SCCL_MAX_NUM_STEPS){
                  WARN("SCCL: too many steps are requested. Max number of steps: %d, requested: %d", SCCL_MAX_NUM_STEPS, s+1);
                  return ncclInternalError;
                }
                if (s < 0){
                  WARN("SCCL: step must be positive: step %d", s);
                  return ncclInternalError;
                }

                int hasSend = 0;
                int hasRecv = 0;
                int checkSrc = 0;
                int checkDst = 0;
                int transferType = -1; // -1 indicate a nop
                if (strcmp(type, "s") == 0){
                  transferType = SCCL_SEND;
                  hasSend = 1;
                  checkSrc = 1;
                } else if (strcmp(type, "r") == 0) {
                  transferType = SCCL_RECV;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rcs") == 0) {
                  transferType = SCCL_RECV_COPY_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rrs") == 0) {
                  transferType = SCCL_RECV_REDUCE_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkSrc = 1;
                } else if (strcmp(type, "rrc") == 0) {
                  transferType = SCCL_RECV_REDUCE_COPY;
                  hasRecv = 1;
                } else if (strcmp(type, "rrcs") == 0) {
                  transferType = SCCL_RECV_REDUCE_COPY_SEND;
                  hasRecv = 1;
                  hasSend = 1;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "cpy") == 0) {
                  transferType = SCCL_LOCAL_COPY;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "re") == 0) {
                  transferType = SCCL_REDUCE;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "res") == 0) {
                  transferType = SCCL_RES_ADD;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "nop") == 0) {
                  transferType = -1;
                } else {
                  WARN("SCCL: type of transfer is not supported: %s", type);
                  return ncclInternalError;
                }

                if (depend_bid >= 0) {
                  sTB->dependentBid[numDependences] = depend_bid;
                  sTB->dependentStep[numDependences] = depend_step;
                  numDependences++;
                }

                uint8_t srcbufferInt = 0;
                uint8_t dstbufferInt = 0;
                NCCLCHECK(scclGetBufferType(srcbuffer, &srcbufferInt));
                NCCLCHECK(scclGetBufferType(dstbuffer, &dstbufferInt));

                int continuationOfReductions = 0;
                // Analyze to see if this is in the same list of reductions for them to be chained
                if (transferType == SCCL_REDUCE && oldReductionDstBuffer == dstbufferInt && oldReductionDstOffset == dstoffset && oldReductionSrcBuffer == srcbufferInt && depend_bid == -1){
                  numTransfers--; // reuse the same transfer
                  continuationOfReductions = 1;
                }


                if (transferType != -1) {
                  struct scclTransfer* sccltran = &sTB->transfers[numTransfers];
                  sccltran->type = transferType;
                  sccltran->srcoffset = srcoffset;
                  sccltran->srcbuffer = srcbufferInt;
                  sccltran->srcoffset = srcoffset;
                  sccltran->dstbuffer = dstbufferInt;
                  sccltran->dstoffset = dstoffset;

                  if (count < 0 || count >= SCCL_MAX_COUNT){
                    WARN("SCCL: count (%d) must be positive and less than %d", count, SCCL_MAX_COUNT);
                    return ncclInternalError;
                  }
                  sccltran->count = count;

                  if (hasSend){
                    if (sendpeer < 0){
                      WARN("SCCL: there is a send in threadblock %d on GPU %d without a sendpeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (scclChannel == NULL) {
                      WARN("SCCL: something went wrong! Channel should not have been NULL on threadblock %d GPU %d.", bid, id);
                      return ncclInternalError;
                    }
                    scclChannel->nchunksForSendPeer[scclChannel->nsendPeers][count-1]++;
                  }
                  if (hasRecv){
                    if (recvpeer < 0){
                      WARN("SCCL: there is a recv in threadblock %d on GPU %d without a recvpeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (scclChannel == NULL) {
                      WARN("SCCL: something went wrong! Channel should not have been NULL on threadblock %d GPU %d.", bid, id);
                      return ncclInternalError;
                    }
                    scclChannel->nchunksForRecvPeer[scclChannel->nrecvPeers][count-1]++;
                  }

                  if (checkSrc) NCCLCHECK(scclCheckBufferBounds(sccltran->srcbuffer, sccltran->srcoffset, nInputChunks, nOutputChunks, nScratchChunks));
                  if (checkDst) NCCLCHECK(scclCheckBufferBounds(sccltran->dstbuffer, sccltran->dstoffset, nInputChunks, nOutputChunks, nScratchChunks));

                  if (!continuationOfReductions){
                    sccltran->depencePointer = oldDependencePointer;
                    sccltran->numDependences = numDependences - oldDependencePointer;
                    if (sccltran->numDependences > 0 && depend_bid < 0){
                      WARN("SCCL: when there is a chain of dependences, the last reduction must be a part of the first immediate instruction. Detected for GPU %d, threadblock %d, and step %d. XML will be ignored.", id, bid, s);
                      return ncclInvalidUsage;
                    }
                    oldDependencePointer = numDependences;
                  }

                  // reduction related pointers
                  if (transferType != SCCL_REDUCE){
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                    oldReductionSrcBuffer = -1;
                  } else {
                    if (oldReductionDstBuffer == -1) { // if this is the first reduction
                      sccltran->reductionPointer = numReductions;
                    }
                    sTB->reductionSrcOffsets[numReductions] = sccltran->srcoffset;
                    numReductions++;
                    sccltran->numReductions = numReductions - sccltran->reductionPointer;

                    if (has_dependence){
                      oldReductionDstBuffer = -1;
                      oldReductionDstOffset = -1;
                    } else {
                      oldReductionDstBuffer = sccltran->dstbuffer;
                      oldReductionDstOffset = sccltran->dstoffset;
                      oldReductionSrcBuffer = sccltran->srcbuffer;
                    }
                  }


                  if (has_dependence != 0 && has_dependence != 1){
                    WARN("SCCL: has_dependence needs to be 0 or 1, but it was %d", has_dependence);
                    return ncclInternalError;
                  }
                  sccltran->has_dependence = has_dependence;

                  numTransfers++;
                  sTB->nsteps = numTransfers;
                }
              }
            }
            if (sTB->sendpeer >= 0){
              if (scclChannel == NULL) {
                WARN("SCCL: something went wrong! Channel should not have been NULL on threadblock %d GPU %d.", bid, id);
                return ncclInternalError;
              }
              scclChannel->sendPeers[scclChannel->nsendPeers] = sTB->sendpeer;
              scclChannel->nsendPeers++;
            }
            if (sTB->recvpeer >= 0){
              if (scclChannel == NULL) {
                WARN("SCCL: something went wrong! Channel should not have been NULL on threadblock %d GPU %d.", bid, id);
                return ncclInternalError;
              }
              scclChannel->recvPeers[scclChannel->nrecvPeers] = sTB->recvpeer;
              scclChannel->nrecvPeers++;
            }
            if (scclChannel) {
              scclChannel->nBlocksForChannel++;
              if (scclChannel->nBlocksForChannel > SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL){
                WARN("SCCL: too many sends/recv per channel. Max allowed %d", SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                return ncclInvalidUsage;
              }
            }
          }
        }
        // make sure that threblocks are in order. Something like 0, 2, 3 is not allowed.
        for (int i = 1; i < SCCL_MAX_NUM_THREAD_BLOCKS; i++){
          if (blockExists[i] == 1 && blockExists[i-1] == 0){
            WARN("SCCL: threadblock %d is missing", i);
            return ncclInvalidUsage;
          }
        }
      }
    }
  }
  free(xml);
  scclAlgo->isValid = true; // all went well, set isValid to true
  return ncclSuccess;
}

ncclResult_t scclGetAllAlgoFromXMLFilesAndSetComm(struct ncclComm* comm, const char* str){
  INFO(NCCL_ENV, "SCCL_XML_FILES set by environment to %s", str);
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ":", &tmpStr);
  comm->numberOfSCCLAlgorithms = 0;
  while (token) {
    if (comm->numberOfSCCLAlgorithms == SCCL_MAX_NUM_ALGOS){
      WARN("SCCL: too many algorithms (%d) specified in environment variable SCCL_XML_FILES. The rest will be ignored.", comm->numberOfSCCLAlgorithms);
      break;
    }
    struct scclAlgorithm* scclAlgo = &comm->scclAlgos[comm->numberOfSCCLAlgorithms];
    if (scclGetAlgoFromXMLAndSetComm(comm, token, scclAlgo) == ncclSuccess){
      comm->numberOfSCCLAlgorithms++;
      INFO(NCCL_INIT, "Parsed SCCL Algorithm %s successfully.", token);
    } else {
      WARN("SCCL: algorithm %s failed to initialize. Will be ignored.", token);
    }
    token = strtok_r(NULL, ",", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

ncclResult_t scclGetAllAlgoFromSCCLConfigAndSetComm(struct ncclComm* comm, const char* str){
  INFO(NCCL_INIT, "SCCL: Parsing config %s", str);
  struct ncclXml* xml;

  comm->scclRegistrations = NULL;
  comm->nScclRegistrations = 0;

  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(scclGetXmlConfigFromFile(str, xml));

  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "sccl_algos", &topNode));

  for (int s=0; s < topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "load") == 0) {
      if (comm->numberOfSCCLAlgorithms == SCCL_MAX_NUM_ALGOS){
        WARN("SCCL: too many algorithms (%d) specified in environment variable SCCL_XML_FILES. The rest will be ignored.", comm->numberOfSCCLAlgorithms);
        break;
      }

      const char *path;
      NCCLCHECK(xmlGetAttrStr(node, "path", &path));

      int hasMinBytes = false;
      NCCLCHECK(xmlAttrExists(node, "minbytes", &hasMinBytes));
      int64_t minBytes = 0;
      if (hasMinBytes) {
        NCCLCHECK(xmlGetAttrInt64_t(node, "minbytes", &minBytes));
      }

      int hasMaxBytes = false;
      NCCLCHECK(xmlAttrExists(node, "maxbytes", &hasMaxBytes));
      int64_t maxBytes = -1; // Represents infinity
      if (hasMaxBytes) {
        NCCLCHECK(xmlGetAttrInt64_t(node, "maxbytes", &maxBytes));
      }

      int hasProtocol = false;
      NCCLCHECK(xmlAttrExists(node, "proto", &hasProtocol));
      const char *protocol = NULL;
      if (hasProtocol) {
        NCCLCHECK(xmlGetAttrStr(node, "proto", &protocol));
      }

      int algoIndex = comm->numberOfSCCLAlgorithms;
      struct scclAlgorithm* scclAlgo = &comm->scclAlgos[algoIndex];
      if (scclGetAlgoFromXMLAndSetComm(comm, path, scclAlgo) == ncclSuccess){
        comm->numberOfSCCLAlgorithms++;
        INFO(NCCL_INIT, "Parsed SCCL Algorithm %s successfully.", path);

        int regIndex = comm->nScclRegistrations++;
        NCCLCHECK(ncclRealloc(&comm->scclRegistrations, comm->nScclRegistrations));
        struct scclRegistration *scclReg = &comm->scclRegistrations[regIndex];
        scclReg->algoIndex = algoIndex;
        scclReg->minBytes = minBytes;
        scclReg->maxBytes = maxBytes;
        NCCLCHECK(scclProtocolStrToId(protocol, &scclReg->protocol));
      } else {
        WARN("SCCL: algorithm %s failed to initialize. Will be ignored.", path);
      }
    }
  }
  free(xml);
  return ncclSuccess;
}

/****************************/
/* External query functions */
/****************************/

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoSetAffinity(struct ncclTopoSystem* system, int rank) {
  struct ncclTopoNode* cpu = NULL, *gpu = NULL;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      gpu = system->nodes[GPU].nodes+g;
      // Find closer CPU
      int cpuIndex = -1, minHops = 0;
      for (int c=0; c<system->nodes[CPU].count; c++) {
        int nHops = system->nodes[GPU].nodes[g].paths[CPU][c].count;
        if (cpuIndex == -1 || nHops < minHops) {
          cpuIndex = c;
          minHops = nHops;
        }
      }
      cpu = system->nodes[CPU].nodes+cpuIndex;
    }
  }
  if (cpu == NULL) {
    WARN("Set CPU affinity : unable to find GPU/CPU for rank %d", rank);
    return ncclInternalError;
  }

  // Query the CPU affinity set we were provided
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&mask, affinityStr));
    TRACE(NCCL_INIT, "Current affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  // Get the affinity of the CPU close to our GPU.
  cpu_set_t cpuMask = cpu->cpu.affinity;

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&cpuMask, affinityStr));
    TRACE(NCCL_INIT, "CPU GPU affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = cpuMask;
  else
    // Use a subset of the GPU affinity set
    CPU_AND(&finalMask, &mask, &cpuMask);

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&finalMask, affinityStr));
    INFO(NCCL_INIT, "Setting affinity for GPU %d to %s", gpu->gpu.dev, affinityStr);
    SYSCHECK(sched_setaffinity(0, sizeof(cpu_set_t), &finalMask), "sched_setaffinity");
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetCompCap(struct ncclTopoSystem* system, int* ccMin, int* ccMax) {
  if (system->nodes[GPU].count == 0) return ncclInternalError;
  int min, max;
  min = max = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  for (int g=1; g<system->nodes[GPU].count; g++) {
    min = std::min(min, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
    max = std::max(max, system->nodes[GPU].nodes[g].gpu.cudaCompCap);
  }
  if (ccMin) *ccMin = min;
  if (ccMax) *ccMax = max;
  return ncclSuccess;
}
