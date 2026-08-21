// probe_metal.mm — 诊断：这台机器的 Metal 运行时对象模型
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <objc/runtime.h>
#include <cstdio>
#include <cstring>

int main() {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  printf("device class: %s\n", object_getClassName(dev));
  id<MTLCommandQueue> q = [dev newCommandQueue];
  printf("queue class: %s\n", object_getClassName(q));
  id cb = [q commandBuffer];
  printf("cb class: %s\n", object_getClassName(cb));
  printf("responds computeEncoder: %d\n",
         [cb respondsToSelector:@selector(computeEncoder)]);
  printf("responds commit: %d\n", [cb respondsToSelector:@selector(commit)]);
  unsigned int n = 0;
  Method* methods = class_copyMethodList(object_getClass(cb), &n);
  int shown = 0;
  for (unsigned int i = 0; i < n && shown < 40; ++i) {
    const char* name = sel_getName(method_getName(methods[i]));
    if (strstr(name, "ncoder") || strstr(name, "ommit")) {
      printf("  method: %s\n", name);
      ++shown;
    }
  }
  free(methods);
  return 0;
}
