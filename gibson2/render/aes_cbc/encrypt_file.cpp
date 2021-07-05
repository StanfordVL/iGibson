// g++ -g3 -ggdb -O0 -DDEBUG -I/usr/include/cryptopp Driver.cpp -o Driver.exe -lcryptopp -lpthread
// g++ -g -O2 -DNDEBUG -I/usr/include/cryptopp Driver.cpp -o Driver.exe -lcryptopp -lpthread

#include "osrng.h"
using CryptoPP::AutoSeededRandomPool;

#include <iostream>
#include <fstream>
#include <sstream>

using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;

#include <string>
using std::string;

#include <cstdlib>
using std::exit;

#include "cryptlib.h"
using CryptoPP::Exception;

#include "hex.h"
using CryptoPP::HexEncoder;
using CryptoPP::HexDecoder;

#include "filters.h"
using CryptoPP::StringSink;
using CryptoPP::StringSource;
using CryptoPP::StreamTransformationFilter;
using CryptoPP::ArraySink;

#include "aes.h"
using CryptoPP::AES;

#include "ccm.h"
using CryptoPP::CBC_Mode;

#include "assert.h"

using CryptoPP::byte;

int main(int argc, char* argv[])
{
	AutoSeededRandomPool prng;

	byte key[AES::DEFAULT_KEYLENGTH];
	byte iv[AES::BLOCKSIZE];
    ifstream key_file(argv[1]);
    string key_string, iv_string;
    std::getline(key_file, key_string);
    std::getline(key_file, iv_string);

    cout << key_string << endl;
    cout << iv_string << endl;

    StringSource(key_string, true,
		new HexDecoder(
			new ArraySink(key, sizeof(key))
		) // HexEncoder
	); // StringSource

    StringSource(iv_string, true,
		new HexDecoder(
			new ArraySink(iv, sizeof(iv))
		) // HexEncoder
	); // StringSource

    char * input_filename = argv[2];
    char * output_filename = argv[3];

    ifstream in_file;
    in_file.open(input_filename);
    stringstream str_stream;
    str_stream << in_file.rdbuf(); //read the file

	string plain = str_stream.str();
	string cipher, encoded, recovered;

	try
	{
		//cout << "plain text: " << plain << endl;

		CBC_Mode< AES >::Encryption e;
		e.SetKeyWithIV(key, sizeof(key), iv);

		// The StreamTransformationFilter removes
		//  padding as required.
		StringSource s(plain, true, 
			new StreamTransformationFilter(e,
				new StringSink(cipher)
			) // StreamTransformationFilter
		); // StringSource

#if 0
		StreamTransformationFilter filter(e);
		filter.Put((const byte*)plain.data(), plain.size());
		filter.MessageEnd();

		const size_t ret = filter.MaxRetrievable();
		cipher.resize(ret);
		filter.Get((byte*)cipher.data(), cipher.size());
#endif
	}
	catch(const CryptoPP::Exception& e)
	{
		cerr << e.what() << endl;
		exit(1);
	}

	ofstream out_file;
    out_file.open(output_filename);
    out_file << cipher;
	//cout << cipher << endl;

	return 0;
}

