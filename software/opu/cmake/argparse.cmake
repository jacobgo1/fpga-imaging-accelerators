Include(FetchContent)

set(ARGPARSE_BUILD_SAMPLES OFF CACHE BOOL "Disable argparse samples" FORCE)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)
