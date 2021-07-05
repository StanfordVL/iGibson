
add_executable(aes_cbc_example Driver.cpp)
target_link_libraries(aes_cbc_example cryptopp-shared)

add_executable(generate_key generate_key.cpp)
target_link_libraries(generate_key cryptopp-shared)

add_executable(encrypt_file encrypt_file.cpp)
target_link_libraries(encrypt_file cryptopp-shared)

add_executable(decrypt_file decrypt_file.cpp)
target_link_libraries(decrypt_file cryptopp-shared)