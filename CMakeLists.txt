cmake_minimum_required(VERSION 3.16)
project(obs-spectrogram VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(libobs REQUIRED)

add_library(obs-spectrogram MODULE
    src/plugin-main.c
    src/spectrogram-source.c
    src/fft.c
)

target_link_libraries(obs-spectrogram
    PRIVATE
    OBS::libobs
)

if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(obs-spectrogram PRIVATE Threads::Threads m)
endif()

target_include_directories(obs-spectrogram PRIVATE src)

set_target_properties(obs-spectrogram PROPERTIES PREFIX "")

include(GNUInstallDirs)

if(WIN32)
    set(OBS_PLUGIN_DEST "obs-plugins/64bit")
    set(OBS_DATA_DEST   "data/obs-plugins/obs-spectrogram")
else()
    set(OBS_PLUGIN_DEST "${CMAKE_INSTALL_LIBDIR}/obs-plugins")
    set(OBS_DATA_DEST   "${CMAKE_INSTALL_DATAROOTDIR}/obs/obs-plugins/obs-spectrogram")
endif()

install(TARGETS obs-spectrogram
    LIBRARY DESTINATION "${OBS_PLUGIN_DEST}"
    RUNTIME DESTINATION "${OBS_PLUGIN_DEST}"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/"
    DESTINATION "${OBS_DATA_DEST}"
)
