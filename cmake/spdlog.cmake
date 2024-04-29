include_guard()
include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.7.0
    SYSTEM
    OVERRIDE_FIND_PACKAGE
)
set(SPDLOG_FMT_EXTERNAL_HO ON)
FetchContent_MakeAvailable(spdlog)
