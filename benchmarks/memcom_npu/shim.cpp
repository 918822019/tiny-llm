#include <cstring>
namespace qnn_wrapper_api {
char *strnDup(const char *source, size_t maxlen) { return ::strndup(source, maxlen); }
}
