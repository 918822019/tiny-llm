# Embed a binary file as an aligned byte array.  The generated header is kept
# in the build tree; source control contains the shader source, not opaque
# generated SPIR-V words.
if (NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
    message(FATAL_ERROR "embed_binary.cmake requires INPUT, OUTPUT and SYMBOL")
endif ()

file(READ "${INPUT}" _hex HEX)
string(REGEX REPLACE "([0-9A-Fa-f][0-9A-Fa-f])" "0x\\1, " _bytes "${_hex}")
file(WRITE "${OUTPUT}"
        "#pragma once\n"
        "#include <cstddef>\n"
        "alignas(4) inline constexpr unsigned char ${SYMBOL}[] = {\n"
        "${_bytes}\n"
        "};\n"
        "inline constexpr std::size_t ${SYMBOL}Size = sizeof(${SYMBOL});\n")
