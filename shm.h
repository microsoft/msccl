#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

static int shm_allocate(int fd, const int shmsize) {
  return posix_fallocate(fd, 0, shmsize);
}
static int shm_map(int fd, const int shmsize, void** ptr) {
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

static void shmSetup(const char* shmname, const int shmsize, int* fd, void** ptr, int create) {
  *fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (create) shm_allocate(*fd, shmsize);
  shm_map(*fd, shmsize, ptr);
  close(*fd);
  *fd = -1;
  if (create) memset(*ptr, 0, shmsize);
}

static void shmOpen(const char* shmname, const int shmsize, void** shmPtr, int create, int cuda) {
  int fd = -1;
  void* ptr = MAP_FAILED;

  shmSetup(shmname, shmsize, &fd, &ptr, create);
  if (cuda) cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped);

  *shmPtr = ptr;
}

static void shmUnlink(const char* shmname) {
  shm_unlink(shmname);
}

static void shmClose(void* shmPtr, const int shmsize, int cuda) {
  if (cuda) cudaHostUnregister(shmPtr);
  munmap(shmPtr, shmsize);
}

