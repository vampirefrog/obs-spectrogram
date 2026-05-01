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
    m
    pthread
)

target_include_directories(obs-spectrogram PRIVATE src)

set_target_properties(obs-spectrogram PROPERTIES PREFIX "")

include(GNUInstallDirs)

install(TARGETS obs-spectrogram
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/obs/obs-plugins/obs-spectrogram"
)
