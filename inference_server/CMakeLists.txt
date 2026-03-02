# SPDX-FileCopyrightText: Copyright (c) 2026 intbot inc.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# intbot inc, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from intbot inc or its affiliates is strictly
# prohibited.

# Build executables for chat, benchmark and accuracy mode

add_executable(llm_http_server llm_http_server.cpp)
target_link_libraries(llm_http_server PRIVATE edgellmCore edgellmTokenizer
                                               exampleUtils commonLibraryExt)
set(HTTP_LIB_HEADER ${CMAKE_SOURCE_DIR}/3rdParty/cpp-httplib/httplib.h)
if(NOT EXISTS ${HTTP_LIB_HEADER})
  message(FATAL_ERROR "cpp-httplib header not found at ${HTTP_LIB_HEADER}. Please add it before building llm_http_server.")
endif()
target_include_directories(
  llm_http_server
  PRIVATE ${COMMON_INCLUDE_DIRS}
          ${CMAKE_SOURCE_DIR}/examples/utils
          ${CMAKE_SOURCE_DIR}/3rdParty/cpp-httplib)
add_cross_build_link_options(llm_http_server)
