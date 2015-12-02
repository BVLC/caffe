## Miscellaneous Node Helpers

 - <a href="#api_nan_make_callback"><b><code>Nan::MakeCallback()</code></b></a>
 - <a href="#api_nan_object_wrap"><b><code>Nan::ObjectWrap</code></b></a>
 - <a href="#api_nan_module_init"><b><code>NAN_MODULE_INIT()</code></b></a>
 - <a href="#api_nan_export"><b><code>Nan::Export()</code></b></a>


<a name="api_nan_make_callback"></a>
### Nan::MakeCallback()

Wrappers around `node::MakeCallback()` providing a consistent API across all supported versions of Node.

Use `MakeCallback()` rather than using `v8::Function#Call()` directly in order to properly process internal Node functionality including domains, async hooks, the microtask queue, and other debugging functionality.

Signatures:

```c++
v8::Local<v8::Value> Nan::MakeCallback(v8::Local<v8::Object> target,
                                       v8::Local<v8::Function> func,
                                       int argc,
                                       v8::Local<v8::Value>* argv);
v8::Local<v8::Value> Nan::MakeCallback(v8::Local<v8::Object> target,
                                       v8::Local<v8::String> symbol,
                                       int argc,
                                       v8::Local<v8::Value>* argv);
v8::Local<v8::Value> Nan::MakeCallback(v8::Local<v8::Object> target,
                                       const char* method,
                                       int argc,
                                       v8::Local<v8::Value>* argv);
```


<a name="api_nan_object_wrap"></a>
### Nan::ObjectWrap()

A reimplementation of `node::ObjectWrap` that adds some API not present in older versions of Node. Should be preferred over `node::ObjectWrap` in all cases for consistency.

See the Node documentation on [Wrapping C++ Objects](https://nodejs.org/api/addons.html#addons_wrapping_c_objects) for more details.

Definition:

```c++
class ObjectWrap {
 public:
  ObjectWrap();

  virtual ~ObjectWrap();

  template <class T>
  static inline T* Unwrap(v8::Local<v8::Object> handle);

  inline v8::Local<v8::Object> handle();

  inline Nan::Persistent<v8::Object>& persistent();

 protected:
  inline void Wrap(v8::Local<v8::Object> handle);

  inline void MakeWeak();

  /* Ref() marks the object as being attached to an event loop.
   * Refed objects will not be garbage collected, even if
   * all references are lost.
   */
  virtual void Ref();

  /* Unref() marks an object as detached from the event loop.  This is its
   * default state.  When an object with a "weak" reference changes from
   * attached to detached state it will be freed. Be careful not to access
   * the object after making this call as it might be gone!
   * (A "weak reference" means an object that only has a
   * persistant handle.)
   *
   * DO NOT CALL THIS FROM DESTRUCTOR
   */
  virtual void Unref();

  int refs_;  // ro
};
```


<a name="api_nan_module_init"></a>
### NAN_MODULE_INIT()

Used to define the entry point function to a Node add-on. Creates a function with a given `name` that receives a `target` object representing the equivalent of the JavaScript `exports` object.

See example below.

<a name="api_nan_export"></a>
### Nan::Export()

A simple helper to register a `v8::FunctionTemplate` from a JavaScript-accessible method (see [Methods](./methods.md)) as a property on an object. Can be used in a way similar to assigning properties to `module.exports` in JavaScript.

Signature:

```c++
void Export(v8::Local<v8::Object> target, const char *name, Nan::FunctionCallback f)
```

Also available as the shortcut `NAN_EXPORT` macro.

Example:

```c++
NAN_METHOD(Foo) {
  ...
}

NAN_MODULE_INIT(Init) {
  NAN_EXPORT(target, Foo);
}
```
