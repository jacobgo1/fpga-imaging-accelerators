Include(FetchContent)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable libconfig examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Disable libconfig tests" FORCE)
set(BUILD_CXX ON CACHE BOOL "Build libconfig C++ bindings" FORCE)

FetchContent_Declare(
    libconfig
    GIT_REPOSITORY https://github.com/hyperrealm/libconfig.git
    GIT_TAG v1.8.2
)
FetchContent_MakeAvailable(libconfig)
