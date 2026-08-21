// bench_nand.c — NAND/闪存顺序读带宽（测量项 5）
//
// 用 F_NOCACHE 绕过 page cache，测闪存顺序读的真实上限
// （flash offload / mmap 权重轴的天花板）。
//
// 用法：./bench_nand <file> [chunk_mb] [passes]
// 编译：clang -O3 bench_nand.c -o bench_nand
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <file> [chunk_mb] [passes]\n", argv[0]);
    return 1;
  }
  const char* path = argv[1];
  size_t chunk_mb = (argc > 2) ? atol(argv[2]) : 4;
  int passes = (argc > 3) ? atoi(argv[3]) : 2;

  int fd = open(path, O_RDONLY);
  if (fd < 0) { perror("open"); return 1; }
  // macOS：绕过 page cache，直接测闪存
  if (fcntl(fd, F_NOCACHE, 1) < 0) perror("F_NOCACHE (continuing)");

  off_t total = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);
  size_t chunk = chunk_mb << 20;
  char* buf = malloc(chunk);
  if (!buf) { fprintf(stderr, "alloc fail\n"); return 1; }

  printf("file_bytes\t%lld\nchunk_MB\t%zu\n", (long long)total, chunk_mb);
  printf("pass\tGB/s\n");
  for (int p = 0; p < passes; ++p) {
    lseek(fd, 0, SEEK_SET);
    uint64_t rd = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ssize_t n;
    while ((n = read(fd, buf, chunk)) > 0) rd += (uint64_t)n;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("%d\t%.2f\n", p, rd / dt / 1e9);
  }
  free(buf);
  close(fd);
  return 0;
}
