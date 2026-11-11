Include(FetchContent)

set(SPDLOG_BUILD_ALL OFF CACHE BOOL "Disable spdlog extras" FORCE)
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "Build spdlog static" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Disable spdlog tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Disable spdlog benchmarks" FORCE)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)
