/* Copyright (c) 2010, Ben Noordhuis <info@bnoordhuis.nl>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "BoyerMoore.h"
#include "node.h"
#include "node_buffer.h"
#include "node_version.h"
#include "v8.h"

#include <algorithm>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace {

using v8::Exception;
using v8::Handle;
using v8::Local;
using v8::Object;
using v8::String;
using v8::Value;

#if NODE_MAJOR_VERSION > 0 || NODE_MINOR_VERSION > 10
# define UNI_BOOLEAN_NEW(value)                                               \
    v8::Boolean::New(args.GetIsolate(), value)
# if NODE_MAJOR_VERSION >= 3
#  define UNI_BUFFER_NEW(size)                                                \
    node::Buffer::New(args.GetIsolate(), size).ToLocalChecked()
# else
#  define UNI_BUFFER_NEW(size)                                                \
    node::Buffer::New(args.GetIsolate(), size)
# endif  // NODE_MAJOR_VERSION >= 3
# define UNI_CONST_ARGUMENTS(name)                                            \
    const v8::FunctionCallbackInfo<v8::Value>& name
# define UNI_ESCAPE(value)                                                    \
    return handle_scope.Escape(value)
# define UNI_ESCAPABLE_HANDLESCOPE()                                          \
    v8::EscapableHandleScope handle_scope(args.GetIsolate())
# define UNI_FUNCTION_CALLBACK(name)                                          \
    void name(const v8::FunctionCallbackInfo<v8::Value>& args)
# define UNI_HANDLESCOPE()                                                    \
    v8::HandleScope handle_scope(args.GetIsolate())
# define UNI_INTEGER_NEW(value)                                               \
    v8::Integer::New(args.GetIsolate(), value)
# define UNI_RETURN(value)                                                    \
    args.GetReturnValue().Set(value)
# define UNI_STRING_EMPTY()                                                   \
    v8::String::Empty(args.GetIsolate())
# define UNI_STRING_NEW(string, size)                                         \
    v8::String::NewFromUtf8(args.GetIsolate(),                                \
                            string,                                           \
                            v8::String::kNormalString,                        \
                            size)
# define UNI_THROW_AND_RETURN(type, message)                                  \
    do {                                                                      \
      args.GetIsolate()->ThrowException(                                      \
          type(v8::String::NewFromUtf8(args.GetIsolate(), message)));         \
      return;                                                                 \
    } while (0)
# define UNI_THROW_EXCEPTION(type, message)                                   \
    args.GetIsolate()->ThrowException(                                        \
        type(v8::String::NewFromUtf8(args.GetIsolate(), message)));
#else  // NODE_MAJOR_VERSION > 0 || NODE_MINOR_VERSION > 10
# define UNI_BOOLEAN_NEW(value)                                               \
    v8::Local<v8::Boolean>::New(v8::Boolean::New(value))
# define UNI_BUFFER_NEW(size)                                                 \
    v8::Local<v8::Object>::New(node::Buffer::New(size)->handle_)
# define UNI_CONST_ARGUMENTS(name)                                            \
    const v8::Arguments& name
# define UNI_ESCAPE(value)                                                    \
    return handle_scope.Close(value)
# define UNI_ESCAPABLE_HANDLESCOPE()                                          \
    v8::HandleScope handle_scope
# define UNI_FUNCTION_CALLBACK(name)                                          \
    v8::Handle<v8::Value> name(const v8::Arguments& args)
# define UNI_HANDLESCOPE()                                                    \
    v8::HandleScope handle_scope
# define UNI_INTEGER_NEW(value)                                               \
    v8::Integer::New(value)
# define UNI_RETURN(value)                                                    \
    return handle_scope.Close(value)
# define UNI_STRING_EMPTY()                                                   \
    v8::String::Empty()
# define UNI_STRING_NEW(string, size)                                         \
    v8::String::New(string, size)
# define UNI_THROW_AND_RETURN(type, message)                                  \
    return v8::ThrowException(v8::String::New(message))
# define UNI_THROW_EXCEPTION(type, message)                                   \
    v8::ThrowException(v8::String::New(message))
#endif  // NODE_MAJOR_VERSION > 0 || NODE_MINOR_VERSION > 10

#if defined(_WIN32)
// Emulate snprintf() on windows, _snprintf() doesn't zero-terminate
// the buffer on overflow.
inline int snprintf(char* buf, size_t size, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  const int len = _vsprintf_p(buf, size, fmt, ap);
  va_end(ap);
  if (len < 0) {
    abort();
  }
  if (static_cast<unsigned>(len) >= size && size > 0) {
    buf[size - 1] = '\0';
  }
  return len;
}
#endif

// this is an application of the Curiously Recurring Template Pattern
template <class Derived> struct UnaryAction {
  Local<Value> apply(Local<Object> buffer,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start);

  Local<Value> operator()(UNI_CONST_ARGUMENTS(args)) {
    UNI_ESCAPABLE_HANDLESCOPE();

    uint32_t args_start = 0;
    Local<Object> target = args.This();
    if (node::Buffer::HasInstance(target)) {
      // Invoked as prototype method, no action required.
    } else if (node::Buffer::HasInstance(args[0])) {
      // First argument is the target buffer.
      args_start = 1;
      target = args[0]->ToObject();
    } else {
      UNI_THROW_EXCEPTION(Exception::TypeError,
                          "Argument should be a buffer object.");
      return Local<Value>();
    }

    UNI_ESCAPE(static_cast<Derived*>(this)->apply(target, args, args_start));
  }
};

template <class Derived> struct BinaryAction {
  Local<Value> apply(Local<Object> buffer,
                     const uint8_t* data,
                     size_t size,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start);

  Local<Value> operator()(UNI_CONST_ARGUMENTS(args)) {
    UNI_ESCAPABLE_HANDLESCOPE();

    uint32_t args_start = 0;
    Local<Object> target = args.This();
    if (node::Buffer::HasInstance(target)) {
      // Invoked as prototype method, no action required.
    } else if (node::Buffer::HasInstance(args[0])) {
      // First argument is the target buffer.
      args_start = 1;
      target = args[0]->ToObject();
    } else {
      UNI_THROW_EXCEPTION(Exception::TypeError,
                          "Argument should be a buffer object.");
      return Local<Value>();
    }

    if (args[args_start]->IsString()) {
      String::Utf8Value s(args[args_start]);
      UNI_ESCAPE(static_cast<Derived*>(this)->apply(
          target,
          (const uint8_t*) *s,
          s.length(),
          args,
          args_start));
    }

    if (node::Buffer::HasInstance(args[args_start])) {
      Local<Object> other = args[args_start]->ToObject();
      UNI_ESCAPE(static_cast<Derived*>(this)->apply(
          target,
          (const uint8_t*) node::Buffer::Data(other),
          node::Buffer::Length(other),
          args,
          args_start));
    }

    UNI_THROW_EXCEPTION(Exception::TypeError,
                        "Second argument must be a string or a buffer.");
    return Local<Value>();
  }
};

//
// helper functions
//
Local<Value> clear(Local<Object> buffer, int c) {
  size_t length = node::Buffer::Length(buffer);
  uint8_t* data = (uint8_t*) node::Buffer::Data(buffer);
  memset(data, c, length);
  return buffer;
}

Local<Value> fill(Local<Object> buffer, void* pattern, size_t size) {
  size_t length = node::Buffer::Length(buffer);
  uint8_t* data = (uint8_t*) node::Buffer::Data(buffer);

  if (size >= length) {
    memcpy(data, pattern, length);
  } else {
    const int n_copies = length / size;
    const int remainder = length % size;
    for (int i = 0; i < n_copies; i++) {
      memcpy(data + size * i, pattern, size);
    }
    memcpy(data + size * n_copies, pattern, remainder);
  }

  return buffer;
}

int compare(Local<Object> buffer, const uint8_t* data2, size_t length2) {
  size_t length = node::Buffer::Length(buffer);
  if (length != length2) {
    return length > length2 ? 1 : -1;
  }

  const uint8_t* data = (const uint8_t*) node::Buffer::Data(buffer);
  return memcmp(data, data2, length);
}

//
// actions
//
struct ClearAction: UnaryAction<ClearAction> {
  Local<Value> apply(Local<Object> buffer,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    return clear(buffer, 0);
  }
};

struct FillAction: UnaryAction<FillAction> {
  Local<Value> apply(Local<Object> buffer,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    if (args[args_start]->IsInt32()) {
      int c = args[args_start]->Int32Value();
      return clear(buffer, c);
    }

    if (args[args_start]->IsString()) {
      String::Utf8Value s(args[args_start]);
      return fill(buffer, *s, s.length());
    }

    if (node::Buffer::HasInstance(args[args_start])) {
      Local<Object> other = args[args_start]->ToObject();
      size_t length = node::Buffer::Length(other);
      uint8_t* data = (uint8_t*) node::Buffer::Data(other);
      return fill(buffer, data, length);
    }

    UNI_THROW_EXCEPTION(Exception::TypeError,
                        "Second argument should be either a string, a buffer "
                        "or an integer.");
    return Local<Value>();
  }
};

struct ReverseAction: UnaryAction<ReverseAction> {
  // O(n/2) for all cases which is okay, might be optimized some more with whole-word swaps
  // XXX won't this trash the L1 cache something awful?
  Local<Value> apply(Local<Object> buffer,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    uint8_t* head = (uint8_t*) node::Buffer::Data(buffer);
    uint8_t* tail = head + node::Buffer::Length(buffer);

    while (head < tail) {
      --tail;
      uint8_t t = *head;
      *head = *tail;
      *tail = t;
      ++head;
    }

    return buffer;
  }
};

struct EqualsAction: BinaryAction<EqualsAction> {
  Local<Value> apply(Local<Object> buffer,
                     const uint8_t* data,
                     size_t size,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    return UNI_BOOLEAN_NEW(compare(buffer, data, size) == 0);
  }
};

struct CompareAction: BinaryAction<CompareAction> {
  Local<Value> apply(Local<Object> buffer,
                     const uint8_t* data,
                     size_t size,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    return UNI_INTEGER_NEW(compare(buffer, data, size));
  }
};

struct IndexOfAction: BinaryAction<IndexOfAction> {
  Local<Value> apply(Local<Object> buffer,
                     const uint8_t* data2,
                     size_t size2,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    const uint8_t* data = (const uint8_t*) node::Buffer::Data(buffer);
    const size_t size = node::Buffer::Length(buffer);

    int32_t start = args[args_start + 1]->Int32Value();

    if (start < 0)
      start = size - std::min<size_t>(size, -start);
    else if (static_cast<size_t>(start) > size)
      start = size;

    const uint8_t* p = boyermoore_search(
      data + start, size - start, data2, size2);

    const ptrdiff_t offset = p ? (p - data) : -1;
    return UNI_INTEGER_NEW(offset);
  }
};

static char toHexTable[] = "0123456789abcdef";

// CHECKME is this cache efficient?
static char fromHexTable[] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1,-1,-1,
    10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
};

inline Local<Value> decodeHex(const uint8_t* const data,
                              const size_t size,
                              UNI_CONST_ARGUMENTS(args),
                              uint32_t args_start) {
  if (size & 1) {
    UNI_THROW_EXCEPTION(Exception::Error,
                        "Odd string length, this is not hexadecimal data.");
    return Local<Value>();
  }

  if (size == 0) {
    return UNI_STRING_EMPTY();
  }

  Local<Object> buffer = UNI_BUFFER_NEW(size / 2);
  uint8_t *src = (uint8_t *) data;
  uint8_t *dst = (uint8_t *) (const uint8_t*) node::Buffer::Data(buffer);

  for (size_t i = 0; i < size; i += 2) {
    int a = fromHexTable[*src++];
    int b = fromHexTable[*src++];

    if (a == -1 || b == -1) {
      UNI_THROW_EXCEPTION(Exception::Error, "This is not hexadecimal data.");
      return Local<Value>();
    }

    *dst++ = b | (a << 4);
  }

  return buffer;
}

struct FromHexAction: UnaryAction<FromHexAction> {
  Local<Value> apply(Local<Object> buffer,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    const uint8_t* data = (const uint8_t*) node::Buffer::Data(buffer);
    size_t length = node::Buffer::Length(buffer);
    return decodeHex(data, length, args, args_start);
  }
};

struct ToHexAction: UnaryAction<ToHexAction> {
  Local<Value> apply(Local<Object> buffer,
                     UNI_CONST_ARGUMENTS(args),
                     uint32_t args_start) {
    const size_t size = node::Buffer::Length(buffer);
    const uint8_t* data = (const uint8_t*) node::Buffer::Data(buffer);

    if (size == 0) {
      return UNI_STRING_EMPTY();
    }

    std::string s(size * 2, 0);
    for (size_t i = 0; i < size; ++i) {
      const uint8_t c = (uint8_t) data[i];
      s[i * 2] = toHexTable[c >> 4];
      s[i * 2 + 1] = toHexTable[c & 15];
    }

    return UNI_STRING_NEW(s.c_str(), s.size());
  }
};

//
// V8 function callbacks
//
#define V(name)                                                               \
  UNI_FUNCTION_CALLBACK(name) {                                               \
    UNI_HANDLESCOPE();                                                        \
    UNI_RETURN(name ## Action()(args));                                       \
  }
V(Clear)
V(Compare)
V(Equals)
V(Fill)
V(FromHex)
V(IndexOf)
V(Reverse)
V(ToHex)
#undef V

UNI_FUNCTION_CALLBACK(Concat) {
  UNI_HANDLESCOPE();

  size_t size = 0;
  for (int index = 0, length = args.Length(); index < length; ++index) {
    Local<Value> arg = args[index];
    if (arg->IsString()) {
      // Utf8Length() because we need the length in bytes, not characters
      size += arg->ToString()->Utf8Length();
    }
    else if (node::Buffer::HasInstance(arg)) {
      size += node::Buffer::Length(arg->ToObject());
    }
    else {
      char errmsg[256];
      snprintf(errmsg,
               sizeof(errmsg),
               "Argument #%lu is neither a string nor a buffer object.",
               static_cast<unsigned long>(index));
      UNI_THROW_AND_RETURN(Exception::TypeError, errmsg);
    }
  }

  Local<Object> buffer = UNI_BUFFER_NEW(size);
  uint8_t* s = (uint8_t*) node::Buffer::Data(buffer);

  for (int index = 0, length = args.Length(); index < length; ++index) {
    Local<Value> arg = args[index];
    if (arg->IsString()) {
      String::Utf8Value v(arg);
      memcpy(s, *v, v.length());
      s += v.length();
    }
    else if (node::Buffer::HasInstance(arg)) {
      Local<Object> b = arg->ToObject();
      const uint8_t* data = (const uint8_t*) node::Buffer::Data(b);
      size_t length = node::Buffer::Length(b);
      memcpy(s, data, length);
      s += length;
    }
    else {
      UNI_THROW_AND_RETURN(Exception::Error,
                           "Congratulations! You have run into a bug: argument "
                           "is neither a string nor a buffer object.  Please "
                           "make the world a better place and report it.");
    }
  }

  UNI_RETURN(buffer);
}

void RegisterModule(Handle<Object> target) {
  NODE_SET_METHOD(target, "clear", Clear);
  NODE_SET_METHOD(target, "compare", Compare);
  NODE_SET_METHOD(target, "concat", Concat);
  NODE_SET_METHOD(target, "equals", Equals);
  NODE_SET_METHOD(target, "fill", Fill);
  NODE_SET_METHOD(target, "fromHex", FromHex);
  NODE_SET_METHOD(target, "indexOf", IndexOf);
  NODE_SET_METHOD(target, "reverse", Reverse);
  NODE_SET_METHOD(target, "toHex", ToHex);
}

} // anonymous namespace

NODE_MODULE(buffertools, RegisterModule)
