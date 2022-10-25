/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
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
const char* topoLinkTypeStr[] = { "LOC", "NVL", "",    "PCI",    "",    "",    "", "SYS", "NET" };
const char* topoPathTypeStr[] = { "LOC", "NVL", "NVB", "PIX", "PXB", "PXN", "PHB", "SYS", "DIS" };

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
    n->net.latency = 0.0;
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

// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  for (int s=0; s<system->nodes[PCI].count; s++) {
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes+s;
    uint64_t device = pciSwitch->pci.device;
    // Only flatten PEX Gen 4 switches in base mode
    if ((device & 0xfffffffffffff000) == 0x1000c0101000a000) {
      // Find sub switches with the same device ID.
      int64_t* subSwIds;
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l=0; l<pciSwitch->nlinks; l++) {
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || sub->pci.device != device) continue;
        // Save sub switch for later
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        memmove(pciSwitch->links+l, pciSwitch->links+l+1, (pciSwitch->nlinks-l-1)*(sizeof(struct ncclTopoLink)));
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        l--;
      }

      for (int s=0; s<subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        int index;
        NCCLCHECK(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index));
        struct ncclTopoNode* sub = system->nodes[PCI].nodes+index;
        // Connect all sub PCI devices to the parent switch
        for (int l=0; l<sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;
          // Add link from parent PCI switch -> PCI device
          memcpy(pciSwitch->links+pciSwitch->nlinks, sub->links+l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          for (int rl=0; rl<remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECK(ncclTopoRemoveNode(system, PCI, index));
      }
      // Set subdevice to 0x0000 to make sure we don't merge this switch again.
      pciSwitch->pci.device = 0x1000c01010000000;
      free(subSwIds);
      // Restart, as system->nodes[PCI].nodes has changed.
      s = 0;
    }
  }
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
  } else if (node->type == PCI) {
    sprintf(line+offset, "%s/%lX (%lx)", topoNodeTypeStr[node->type], node->id, node->pci.device);
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
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "speed", &mbps, 0));
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.width = mbps / 8000.0;
  if (xmlGetAttrFloat(xmlNet, "latency", &net->net.latency) != ncclSuccess) net->net.latency = 0;
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "port", &net->net.port, 0));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "gdr", &net->net.gdrSupport, 0));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "maxconn", &net->net.maxChannels, MAXCHANNELS));
  NCCLCHECK(xmlGetAttrIntDefault(xmlNet, "coll", &net->net.collSupport, 0));
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
struct kvDict kvDictPciGen[] = {
  { "2.5 GT/s", 15 }, { "5 GT/s", 30 }, { "8 GT/s", 60 }, { "16 GT/s", 120 }, { "32 GT/s", 240 }, /* Kernel 5.6 and earlier */
  { "2.5 GT/s PCIe", 15 }, { "5.0 GT/s PCIe", 30 }, { "8.0 GT/s PCIe", 60 }, { "16.0 GT/s PCIe", 120 }, { "32.0 GT/s PCIe", 240 }, { "64.0 GT/s PCIe", 480 },
  { NULL, 60 /* Default fallback */ } }; // x100 Mbps per lane
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
    NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 48;
    NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 32;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 16;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0);

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

  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
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
static ncclResult_t xmlInitAttrFloat(struct ncclXmlNode* node, const char* attrName, const float value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%f", value);
  }
  return ncclSuccess;
}


ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system) {
  struct ncclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  char* xmlTopoFile = getenv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "NCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xml, 1));
  } else {
    // Try default XML topology location
    NCCLCHECK(ncclTopoGetXmlFromFile("/var/run/nvidia-topologyd/virtualTopology.xml", xml, 0));
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
  if (collNetSupport()) {
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
    NCCLCHECK(xmlInitAttrFloat(netNode, "latency", props.latency));
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

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int* id) {
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
  if (count == 0) {
    *id = -1;
    free(nets);
    return ncclSuccess;
  }

  int rr = system->nodes[GPU].nodes[g].gpu.dev;
  *id = nets[rr%count];
  free(nets);
  return ncclSuccess;
}

/*******************************/
/* MSCCL XML parsing functions */
/*******************************/

ncclResult_t mscclGetBufferType(const char* str, uint8_t* output){
  if (strcmp(str, "i") == 0){
    *output = MSCCL_INPUT_BUFFER;
  } else if (strcmp(str, "o") == 0) {
    *output = MSCCL_OUTPUT_BUFFER;
  } else if (strcmp(str, "s") == 0) {
    *output = MSCCL_SCRATCH_BUFFER;
  } else {
    WARN("type of buffer is not supported: %s", str);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclCheckBufferBounds(int bufferType, int offset, int nInputChunks, int nOutputChunks, int nScratchChunks){
  if (bufferType == MSCCL_INPUT_BUFFER){
    if (offset < -1 || offset >= nInputChunks){
      WARN("Incorrect offset set for input buffer: offset: %d maximum allowed: %d", offset, nInputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == MSCCL_OUTPUT_BUFFER){
    if (offset < -1 || offset >= nOutputChunks){
      WARN("Incorrect offset set for output buffer: offset: %d maximum allowed: %d", offset, nOutputChunks);
      return ncclInvalidUsage;
    }
  } else if (bufferType == MSCCL_SCRATCH_BUFFER){
    if (offset < -1 || offset >= nScratchChunks){
      WARN("Incorrect offset set for scratch buffer: offset: %d maximum allowed: %d", offset, nScratchChunks);
      return ncclInvalidUsage;
    }
  }
  return ncclSuccess;
}

ncclResult_t mscclProtocolStrToId(const char *protocol, int *protocolId) {
  if (strcmp(protocol, "Simple") == 0){
    *protocolId = NCCL_PROTO_SIMPLE;
  } else if (strcmp(protocol, "LL128") == 0){
    *protocolId = NCCL_PROTO_LL128;
  } else if (strcmp(protocol, "LL") == 0){
    *protocolId = NCCL_PROTO_LL;
  } else {
    WARN("MSCCL: protocol %s is not supported.", protocol);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclGetAlgoFromXMLAndSetAlgo(const char* str, struct mscclAlgorithm* mscclAlgo, int maxNChannels, int rank, int nRanks) {
  INFO(NCCL_INIT, "MSCCL: Parsing algorithm %s", str);
  struct ncclXml* xml;
  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(mscclGetXmlAlgoFromFile(str, xml, rank));

  // zeroing out all entries.
  memset(mscclAlgo, 0, sizeof(struct mscclAlgorithm));
  mscclAlgo->isValid = false; // set isValid to false until we hit the return ncclSuccess.
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "algo", &topNode));
  const char* name;
  NCCLCHECK(xmlGetAttrStr(topNode, "name", &name));
  strncpy(mscclAlgo->name, name, MSCCL_MAX_ALGO_NAME);

  int ngpus;
  NCCLCHECK(xmlGetAttrInt(topNode, "ngpus", &ngpus));
  if (nRanks != ngpus){
    WARN("MSCCL: ngpus set in the MSCCL algo (%d) doesn't match the communicator ngpus (%d)", ngpus, nRanks);
    return ncclInvalidUsage;
  }
  mscclAlgo->ngpus = ngpus;
  int nchunksPerLoop;
  NCCLCHECK(xmlGetAttrInt(topNode, "nchunksperloop", &nchunksPerLoop));
  int globalNChannels;
  NCCLCHECK(xmlGetAttrInt(topNode, "nchannels", &globalNChannels));

  const char* protocol;
  NCCLCHECK(xmlGetAttrStr(topNode, "proto", &protocol));
  NCCLCHECK(mscclProtocolStrToId(protocol, &mscclAlgo->protocol));

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
    maxBytes = (((int64_t)1)<<27); // set max to 128 MB which is sufficient for now.
  }
  if (minBytes > maxBytes) {
    WARN("MSCCL: minBytes cannot be greater than maxBytes.");
    return ncclInvalidUsage;
  }
  if (minBytes < 0) {
    WARN("MSCCL: minBytes cannot be negative.");
    return ncclInvalidUsage;
  }
  if (maxBytes < 0) {
    WARN("MSCCL: maxBytes cannot be negative.");
    return ncclInvalidUsage;
  }
  mscclAlgo->minBytes = minBytes;
  mscclAlgo->maxBytes = maxBytes;

  const char* collectiveType;
  NCCLCHECK(xmlGetAttrStr(topNode, "coll", &collectiveType));
  int inputNChunksMultiplier = 1;
  int outputNChunksMultiplier = 1;
  if (strcmp(collectiveType, "allreduce") == 0){
    mscclAlgo->collectiveType = ncclFuncAllReduce;
  } else if (strcmp(collectiveType, "allgather") == 0){
    mscclAlgo->collectiveType = ncclFuncAllGather;
    inputNChunksMultiplier = nRanks;
  } else if (strcmp(collectiveType, "reduce") == 0){
    mscclAlgo->collectiveType = ncclFuncReduce;
  } else if (strcmp(collectiveType, "broadcast") == 0){
    mscclAlgo->collectiveType = ncclFuncBroadcast;
  } else if (strcmp(collectiveType, "alltoall") == 0){
    mscclAlgo->collectiveType = ncclFuncAllToAll;
  } else if (strcmp(collectiveType, "reduce_scatter") == 0){
    mscclAlgo->collectiveType = ncclFuncReduceScatter;
    outputNChunksMultiplier = nRanks;
  } else if (strcmp(collectiveType, "custom") == 0){
    mscclAlgo->collectiveType = ncclFuncCustomCollective;
  } else {
    WARN("MSCCL: collective type %s is not supported.", collectiveType);
    return ncclInvalidUsage;
  }

  int inplace;
  NCCLCHECK(xmlGetAttrInt(topNode, "inplace", &inplace));
  if (inplace) {
    mscclAlgo->inPlace = 1;
  } else {
    mscclAlgo->inPlace = 0;
  }
  int nThreads, hasnthreads;
  nThreads = 0; // default value
  NCCLCHECK(xmlAttrExists(topNode, "nthreads", &hasnthreads));
  if (hasnthreads){
    NCCLCHECK(xmlGetAttrInt(topNode, "nthreads", &nThreads));
    if ((nThreads % WARP_SIZE) != 0){
      WARN("MSCCL nthreads must be a multiplication of %d", WARP_SIZE);
      return ncclInvalidUsage;
    }
  }
  mscclAlgo->nThreads = nThreads;

  if (globalNChannels > maxNChannels){
    WARN("MSCCL: number of desired channels (%d) is more than possible ones (%d)", globalNChannels, maxNChannels);
  }
  mscclAlgo->nChannels = globalNChannels;
  mscclAlgo->nchunksPerLoop  = nchunksPerLoop;
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "gpu") == 0){
      int blockExists[MSCCL_MAX_NUM_THREAD_BLOCKS];
      memset(blockExists, 0, sizeof(int[MSCCL_MAX_NUM_THREAD_BLOCKS]));
      int id, nScratchChunks, nInputChunks, nOutputChunks;
      NCCLCHECK(xmlGetAttrInt(node, "id", &id));
      if (id == rank){
        NCCLCHECK(xmlGetAttrInt(node, "i_chunks", &nInputChunks));
        NCCLCHECK(xmlGetAttrInt(node, "o_chunks", &nOutputChunks));
        NCCLCHECK(xmlGetAttrInt(node, "s_chunks", &nScratchChunks));
        if (nScratchChunks < 0){
          WARN("MSCCL: nScratchChunks must be not negative. nScratchChunks: %d", nScratchChunks);
          return ncclInvalidUsage;
        }
        if ((nInputChunks > 0 && nInputChunks*inputNChunksMultiplier != nchunksPerLoop) || (nOutputChunks > 0 && nOutputChunks*outputNChunksMultiplier != nchunksPerLoop)){
          WARN("Inconsistency between i_chunks/o_chunks (%d/%d) and nchunksperloop (%d) for collective %s", nInputChunks, nOutputChunks, nchunksPerLoop, collectiveType);
          return ncclInvalidUsage;
        }
        mscclAlgo->nScratchChunks = nScratchChunks;
        for (int t=0; t<node->nSubs; t++) {
          struct ncclXmlNode* threadblockNode = node->subs[t];
          if (strcmp(threadblockNode->name, "tb") == 0){
            int bid, recvpeer, sendpeer, recvtype, sendtype, channelId;
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "id", &bid));
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "recv", &recvpeer));
            NCCLCHECK(xmlGetAttrInt(threadblockNode, "send", &sendpeer));

            int hasrtype;
            NCCLCHECK(xmlAttrExists(topNode, "rtype", &hasrtype));
            if (hasrtype) {
	      const char* rtype;
              NCCLCHECK(xmlGetAttrStr(threadblockNode, "rtype", &rtype));
              if (strcmp(rtype, "p2p") == 0){
	        recvtype = 1; // 1 for p2p
	      } else if (strcmp(rtype, "shm") == 0){
	        recvtype = 2; // 2 for shm
	      } if (strcmp(rtype, "nic") == 0){
	        recvtype = 3; // 3 for nic
	      } else {
                WARN("Invalid connection type was used: %s", rtype);
                return ncclInvalidUsage;
	      }
            } else {
              recvtype = 0;
            }

            int hasstype;
            NCCLCHECK(xmlAttrExists(topNode, "stype", &hasstype));
            if (hasstype) {
	      const char* stype;
              NCCLCHECK(xmlGetAttrStr(threadblockNode, "stype", &stype));
              if (strcmp(stype, "p2p") == 0){
	        sendtype = 1; // 1 for p2p
	      } else if (strcmp(stype, "shm") == 0){
	        sendtype = 2; // 2 for shm
	      } if (strcmp(stype, "nic") == 0){
	        sendtype = 3; // 3 for nic
	      } else {
                WARN("Invalid connection type was used: %s", stype);
                return ncclInvalidUsage;
	      }
            } else {
              sendtype = 0;
            }

            NCCLCHECK(xmlGetAttrInt(threadblockNode, "chan", &channelId));
            if (bid < 0){
              WARN("MSCCL: bid must be not negative. bid: %d", bid);
              return ncclInvalidUsage;
            }              
            if (bid >= MSCCL_MAX_NUM_THREAD_BLOCKS){
              WARN("MSCCL: too many thread blocks are requested. Max thread blocks: %d", MSCCL_MAX_NUM_THREAD_BLOCKS);
              return ncclInvalidUsage;
            }
            if (blockExists[bid]){
              WARN("MSCCL: duplicate thread block id %d for MSCCL", bid);
              return ncclInvalidUsage;
            }
            blockExists[bid] = 1;

            if (recvpeer == id || sendpeer == id){
              WARN("MSCCL: peer (%d,%d) and gpu id (%d) must be different", recvpeer, sendpeer, id);
              return ncclInvalidUsage;
            }
            struct mscclThreadBlock* sTB = &mscclAlgo->mscclTBs[bid];
            sTB->nsteps = 0;
            if (recvpeer < -1 || sendpeer < -1){
              WARN("MSCCL: wrong recvpeer (%d) or sendpeer (%d) in threadblock %d on gpu %d", recvpeer, sendpeer, bid, id);
              return ncclInvalidUsage;
            }

            if (recvpeer == id || sendpeer == id){
              WARN("MSCCL: recvpeer (%d) or sendpeer (%d) for threadblock %d cannot be gpu %d", recvpeer, sendpeer, bid, id);
              return ncclInvalidUsage;
            }

            if (recvpeer >= ngpus || sendpeer >= ngpus) {
              WARN("MSCCL: recvpeer (%d) or sendpeer (%d) must be -1 or between 0 and ngpus (%d)", recvpeer, sendpeer, ngpus);
              return ncclInvalidUsage;
            }

            sTB->recvpeer = recvpeer;
            sTB->sendpeer = sendpeer;
            if (channelId < 0 || channelId > MAXCHANNELS){
              WARN("MSCCL: threadblock %d on GPU %d has an invalid channel %d", bid, id, channelId);
              return ncclInvalidUsage;
            }
            sTB->channelId = channelId;

            // setting the summary of the msccl aglorithm in msccl channels
            mscclChannelInfo* mscclChannel = &mscclAlgo->mscclChannels[sTB->channelId];

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

                if (s >= MSCCL_MAX_NUM_STEPS){
                  WARN("MSCCL: too many steps are requested. Max number of steps: %d, requested: %d", MSCCL_MAX_NUM_STEPS, s+1);
                  return ncclInternalError;
                }
                if (s < 0){
                  WARN("MSCCL: step must be positive: step %d", s);
                  return ncclInternalError;
                }

                int hasSend = 0;
                int hasRecv = 0;
                int checkSrc = 0;
                int checkDst = 0;
                int transferType = -1; // -1 indicate a nop
                if (strcmp(type, "s") == 0){
                  transferType = MSCCL_SEND;
                  hasSend = 1;
                  checkSrc = 1;
                } else if (strcmp(type, "r") == 0) {
                  transferType = MSCCL_RECV;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rcs") == 0) {
                  transferType = MSCCL_RECV_COPY_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkDst = 1;
                } else if (strcmp(type, "rrs") == 0) {
                  transferType = MSCCL_RECV_REDUCE_SEND;
                  hasSend = 1;
                  hasRecv = 1;
                  checkSrc = 1;
                } else if (strcmp(type, "rrc") == 0) {
                  transferType = MSCCL_RECV_REDUCE_COPY;
                  hasRecv = 1;
                } else if (strcmp(type, "rrcs") == 0) {
                  transferType = MSCCL_RECV_REDUCE_COPY_SEND;
                  hasRecv = 1;
                  hasSend = 1;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "cpy") == 0) {
                  transferType = MSCCL_LOCAL_COPY;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "re") == 0) {
                  transferType = MSCCL_REDUCE;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "ra") == 0) {
                  transferType = MSCCL_RES_ADD;
                  checkSrc = 1;
                  checkDst = 1;
                } else if (strcmp(type, "nop") == 0) {
                  transferType = -1;
                } else {
                  WARN("MSCCL: type of transfer is not supported: %s", type);
                  return ncclInternalError;
                }

                if (depend_bid >= 0) {
                  sTB->dependentBid[numDependences] = depend_bid;
                  sTB->dependentStep[numDependences] = depend_step;
                  numDependences++;
                }

                uint8_t srcbufferInt = 0;
                uint8_t dstbufferInt = 0;
                NCCLCHECK(mscclGetBufferType(srcbuffer, &srcbufferInt));
                NCCLCHECK(mscclGetBufferType(dstbuffer, &dstbufferInt));

                int continuationOfReductions = 0;
                // Analyze to see if this is in the same list of reductions for them to be chained
                if (transferType == MSCCL_REDUCE) {
                  if (oldReductionDstBuffer == dstbufferInt && oldReductionDstOffset == dstoffset && oldReductionSrcBuffer == srcbufferInt && depend_bid == -1){
                    numTransfers--; // reuse the same transfer
                    continuationOfReductions = 1;
                  } else {
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                  }
                }


                if (transferType != -1) {
                  struct mscclTransfer* msccltran = &sTB->transfers[numTransfers];
                  msccltran->type = transferType;
                  msccltran->srcoffset = srcoffset;
                  msccltran->srcbuffer = srcbufferInt;
                  msccltran->srcoffset = srcoffset;
                  msccltran->dstbuffer = dstbufferInt;
                  msccltran->dstoffset = dstoffset;

                  if (count < 0 || count >= MSCCL_MAX_COUNT){
                    WARN("MSCCL: count (%d) must be positive and less than %d", count, MSCCL_MAX_COUNT);
                    return ncclInternalError;
                  }
                  msccltran->count = count;

                  if (hasSend){
                    if (sendpeer < 0){
                      WARN("MSCCL: there is a send in threadblock %d on GPU %d without a sendpeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (mscclChannel->nSendPeers >= MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL){
                      WARN("MSCCL: too many sends per channel. Max allowed %d", MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                      return ncclInvalidUsage;
                    }

                    struct mscclChannelPeerInfo* sendPeerInfo = &mscclChannel->sendPeerInfo[mscclChannel->nSendPeers];
                    sendPeerInfo->nchunksForPeer[count-1]++;
                    // mscclChannel->nchunksForSendPeer[mscclChannel->nsendPeers][count-1]++;
                  }
                  if (hasRecv){
                    if (recvpeer < 0){
                      WARN("MSCCL: there is a recv in threadblock %d on GPU %d without a recvpeer.", bid, id);
                      return ncclInvalidUsage;
                    }
                    if (mscclChannel->nRecvPeers >= MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL){
                      WARN("MSCCL: too many recvs per channel. Max allowed %d", MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL);
                      return ncclInvalidUsage;
                    }
                    struct mscclChannelPeerInfo* recvPeerInfo = &mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers];
                    recvPeerInfo->nchunksForPeer[count-1]++;
                    // mscclChannel->nchunksForRecvPeer[mscclChannel->nrecvPeers][count-1]++;
                  }

                  if (checkSrc) NCCLCHECK(mscclCheckBufferBounds(msccltran->srcbuffer, msccltran->srcoffset, nInputChunks, nOutputChunks, nScratchChunks));
                  if (checkDst) NCCLCHECK(mscclCheckBufferBounds(msccltran->dstbuffer, msccltran->dstoffset, nInputChunks, nOutputChunks, nScratchChunks));

                  if (!continuationOfReductions){
                    msccltran->depencePointer = oldDependencePointer;
                    msccltran->numDependences = numDependences - oldDependencePointer;
                    if (msccltran->numDependences > 0 && depend_bid < 0){
                      WARN("MSCCL: when there is a chain of dependences, the last reduction must be a part of the first immediate instruction. Detected for GPU %d, threadblock %d, and step %d. XML will be ignored.", id, bid, s);
                      return ncclInvalidUsage;
                    }
                    oldDependencePointer = numDependences;
                  }

                  // reduction related pointers
                  if (transferType != MSCCL_REDUCE){
                    oldReductionDstBuffer = -1;
                    oldReductionDstOffset = -1;
                    oldReductionSrcBuffer = -1;
                  } else {
                    if (oldReductionDstBuffer == -1) { // if this is the first reduction
                      msccltran->reductionPointer = numReductions;
                    }
                    sTB->reductionSrcOffsets[numReductions] = msccltran->srcoffset;
                    numReductions++;
                    msccltran->numReductions = numReductions - msccltran->reductionPointer;

                    if (has_dependence || numReductions == MSCCL_MAX_REDUCE_FUSION){
                      oldReductionDstBuffer = -1;
                      oldReductionDstOffset = -1;
                    } else {
                      oldReductionDstBuffer = msccltran->dstbuffer;
                      oldReductionDstOffset = msccltran->dstoffset;
                      oldReductionSrcBuffer = msccltran->srcbuffer;
                    }
                  }


                  if (has_dependence != 0 && has_dependence != 1){
                    WARN("MSCCL: has_dependence needs to be 0 or 1, but it was %d", has_dependence);
                    return ncclInternalError;
                  }
                  msccltran->has_dependence = has_dependence;

                  numTransfers++;
                  sTB->nsteps = numTransfers;
                }
              }
            }

            // finish up mscclChannel calculation

            for (int c = 0; c < MSCCL_MAX_COUNT; c++){
              struct mscclChannelPeerInfo* sendPeer = &mscclChannel->sendPeerInfo[mscclChannel->nSendPeers];
              if (sendPeer->nchunksForPeer[c] > 0){
                sendPeer->counts[sendPeer->nCountExists] = c;
                sendPeer->nCountExists++;
              }
              struct mscclChannelPeerInfo* recvPeer = &mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers];
              if (recvPeer->nchunksForPeer[c] > 0){
                recvPeer->counts[recvPeer->nCountExists] = c;
                recvPeer->nCountExists++;
              }
            }

            if (sTB->sendpeer >= 0){
              mscclChannel->sendPeerInfo[mscclChannel->nSendPeers].peer = sTB->sendpeer;
              mscclChannel->sendPeerInfo[mscclChannel->nSendPeers].connType = sendtype;
              mscclChannel->nSendPeers++;
            }
            if (sTB->recvpeer >= 0){
              mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers].peer = sTB->recvpeer;
              mscclChannel->recvPeerInfo[mscclChannel->nRecvPeers].connType = recvtype;
              mscclChannel->nRecvPeers++;
            }
          }
        }
        // make sure that threblocks are in order. Something like 0, 2, 3 is not allowed.
        if (blockExists[0] == 1){
          mscclAlgo->nBlocks = 1;
        }
        for (int i = 1; i < MSCCL_MAX_NUM_THREAD_BLOCKS; i++){
          if (blockExists[i] == 1 && blockExists[i-1] == 0){
            WARN("MSCCL: threadblock %d is missing", i);
            return ncclInvalidUsage;
          }
          if (blockExists[i] == 1){
            mscclAlgo->nBlocks = i+1;
          }
        }

      }
    }
  }
  free(xml);
  mscclAlgo->isValid = true; // all went well, set isValid to true
  return ncclSuccess;
}

ncclResult_t mscclGetAllAlgoFromXMLFilesAndSetInfo(const char* str, struct mscclHostCommInfo* mscclInfo, int maxNChannels, int rank, int nRanks){
  INFO(NCCL_ENV, "MSCCL_XML_FILES set by environment to %s", str);
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ":", &tmpStr);
  mscclInfo->numberOfMSCCLAlgorithms = 0;
  while (token) {
    if (mscclInfo->numberOfMSCCLAlgorithms == MSCCL_MAX_NUM_ALGOS){
      WARN("MSCCL: too many algorithms (%d) specified in environment variable MSCCL_XML_FILES. The rest will be ignored.", mscclInfo->numberOfMSCCLAlgorithms);
      break;
    }
    struct mscclAlgorithm* mscclAlgo = &mscclInfo->mscclDevComm.mscclAlgos[mscclInfo->numberOfMSCCLAlgorithms];
    if (mscclGetAlgoFromXMLAndSetAlgo(token, mscclAlgo, maxNChannels, rank, nRanks) == ncclSuccess){
      mscclInfo->numberOfMSCCLAlgorithms++;
      INFO(NCCL_INIT, "Parsed MSCCL Algorithm %s successfully.", token);
    } else {
      WARN("MSCCL: algorithm %s failed to initialize. Will be ignored.", token);
    }
    token = strtok_r(NULL, ":", &tmpStr);
  }
  free(tokStr);
  return ncclSuccess;
}

ncclResult_t mscclGetAllAlgoFromConfigAndSetInfo(const char* str, struct mscclHostCommInfo* mscclInfo, int maxNChannels, int rank, int nRanks){
  INFO(NCCL_INIT, "MSCCL: Parsing config %s", str);
  struct ncclXml* xml;

  mscclInfo->mscclRegistrations = NULL;
  mscclInfo->nMscclRegistrations = 0;

  NCCLCHECK(ncclCalloc(&xml, 1));
  NCCLCHECK(mscclGetXmlConfigFromFile(str, xml));

  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "msccl_algos", &topNode));

  for (int s=0; s < topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "load") == 0) {
      if (mscclInfo->numberOfMSCCLAlgorithms == MSCCL_MAX_NUM_ALGOS){
        WARN("MSCCL: too many algorithms (%d) specified in environment variable MSCCL_XML_FILES. The rest will be ignored.", mscclInfo->numberOfMSCCLAlgorithms);
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

      int algoIndex = mscclInfo->numberOfMSCCLAlgorithms;
      struct mscclAlgorithm* mscclAlgo = &mscclInfo->mscclDevComm.mscclAlgos[algoIndex];
      if (mscclGetAlgoFromXMLAndSetAlgo(path, mscclAlgo, maxNChannels, rank, nRanks) == ncclSuccess){
        mscclInfo->numberOfMSCCLAlgorithms++;
        INFO(NCCL_INIT, "Parsed MSCCL Algorithm %s successfully.", path);

        int regIndex = mscclInfo->nMscclRegistrations++;
        NCCLCHECK(ncclRealloc(&mscclInfo->mscclRegistrations, mscclInfo->nMscclRegistrations-1, mscclInfo->nMscclRegistrations));
        struct mscclRegistration *mscclReg = &mscclInfo->mscclRegistrations[regIndex];
        mscclReg->algoIndex = algoIndex;
        mscclReg->minBytes = minBytes;
        mscclReg->maxBytes = maxBytes;
        NCCLCHECK(mscclProtocolStrToId(protocol, &mscclReg->protocol));
      } else {
        WARN("MSCCL: algorithm %s failed to initialize. Will be ignored.", path);
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

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity) {
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

  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&finalMask, affinityStr));
    INFO(NCCL_INIT, "Setting affinity for GPU %d to %s", gpu->gpu.dev, affinityStr);
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

ncclResult_t ncclTopoGetLocalRank(struct ncclTopoSystem* system, int rank, int* localRank) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *localRank = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find local GPU with rank %d\n", rank);
  return ncclInternalError;
}
