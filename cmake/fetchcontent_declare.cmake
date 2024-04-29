include(FetchContent)

function(FetchContent_Declare_GH name project version)
    if(FETCHCONTENT_PREFER_GIT)
        FetchContent_Declare(
            ${name}
            GIT_REPOSITORY https://github.com/${project}.git
            GIT_TAG ${version}
        )
    else()
        FetchContent_Declare(
            ${name}
            URL https://github.com/${project}/archive/${version}.zip
        )
    endif()
endfunction()

function(FetchContent_Declare_GH_patch name project version patch_command)
    if(FETCHCONTENT_PREFER_GIT)
        FetchContent_Declare(
            ${name}
            GIT_REPOSITORY https://github.com/${project}.git
            GIT_TAG ${version}
            PATCH_COMMAND ${patch_command}
        )
    else()
        FetchContent_Declare(
            ${name}
            URL https://github.com/${project}/archive/${version}.zip
            PATCH_COMMAND ${patch_command}
        )
    endif()
endfunction()

function(FetchContent_Declare_GL name project version)
    if(FETCHCONTENT_PREFER_GIT)
        FetchContent_Declare(
            ${name}
            GIT_REPOSITORY https://gitlab.com/${project}.git
            GIT_TAG ${version}
        )
    else()
        FetchContent_Declare(
            ${name}
            URL https://gitlab.com/${project}/-/archive/${version}/.zip
        )
    endif()
endfunction()
