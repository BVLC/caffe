#ifndef CAFFE_DYNAMIC_LIBRARY_HPP
#define CAFFE_DYNAMIC_LIBRARY_HPP

#include <string>
#include <vector>

namespace caffe {

/**
 * @brief Represents a dynamic library opened at runtime.
 *
 * DynamicLIbrary objects can not be copied but they can be moved.
 * When the object goes out of scope, the underlying handle is automatically
 * closed. Other handles to the same library remain valid this happens.
 */
class DynamicLibrary {
 private:
  /// The raw handle to the opened library as returned by the OS.
  void * handle_;

  /// The path of the library.
  std::string path_;

 public:
  /**
   * @brief Construct an invalid dynamic library.
   */
  DynamicLibrary() : handle_{nullptr} {}

  /**
   * @brief Open a dynamic library by name.
   */
  explicit DynamicLibrary(std::string const & name);

  DynamicLibrary(DynamicLibrary const &) = delete;
  DynamicLibrary(DynamicLibrary && other);

  /**
   * @brief Close the underlying handle.
   *
   * Other handles to the same library remain valid.
   */
  ~DynamicLibrary();

  /**
   * @brief The path of the loaded library as passed to the constructor.
   */
  std::string const & path() const { return path_; }

  /**
   * @brief Find a symbol by name from an opened library.
   *
   * @param name The name of the symbol to look for.
   *
   * @return The found symbol or a nullptr if the symbol is not found.
   */
  void * FindSymbol(std::string const & name) const;

  /**
   * @brief Check if the library is successfully opened.
   *
   * @return True if the library is successfully opened.
   */
  explicit operator bool() const { return handle_; }

  /**
   * @brief Check if the library is successfully opened.
   *
   * @return True if the library is successfully opened.
   */
  bool IsValid() const { return static_cast<bool>(*this); }
};

/**
 * @brief Search for a library in a given search path.
 *
 * The paths are searched in order. Once a library is found,
 * the search is aborted and the library is returned.
 *
 * @param name The name of the libary excluding platform specific prefix and suffix.
 * @param search_path The paths where to search for the library.
 *
 * @return The found library, or an invalid library if no library was found.
 */
DynamicLibrary FindLibrary(
  std::string const & name,
  std::vector<std::string> search_path
);

}  // namespace caffe

#endif  // CAFFE_DYNAMIC_LIBRARY_HPP
