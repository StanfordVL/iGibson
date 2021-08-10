// Use double precision for better python integration.
// Need also define this in `binding.cc`(and all compilation units)
#define TINYOBJLOADER_USE_DOUBLE

#include "tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

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
using namespace tinyobj;

bool tinyobj::LoadObjWithKey(attrib_t *attrib, std::vector<shape_t> *shapes,
             std::vector<material_t> *materials, std::string *warn,
             std::string *err, const char *filename, const char *key_filename,
             const char *mtl_basedir,
             bool trianglulate, bool default_vcols_fallback) {
    attrib->vertices.clear();
    attrib->normals.clear();
    attrib->texcoords.clear();
    attrib->colors.clear();
    shapes->clear();

    byte key[AES::DEFAULT_KEYLENGTH];
    byte iv[AES::BLOCKSIZE];
    std::ifstream key_file(key_filename);
    std::string key_string, iv_string;
    std::getline(key_file, key_string);
    std::getline(key_file, iv_string);

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

  std::ifstream ifs(filename, std::ios::binary);
  std::stringstream str_stream;
  str_stream << ifs.rdbuf(); //read the file

  std::string cipher = str_stream.str();
  std::string plain, encoded, recovered;

	try
	{
		CBC_Mode< AES >::Decryption d;
		d.SetKeyWithIV(key, sizeof(key), iv);

		// The StreamTransformationFilter removes
		//  padding as required.
		StringSource s(cipher, true,
			new StreamTransformationFilter(d,
				new StringSink(recovered)
			) // StreamTransformationFilter
		); // StringSource

#if 0
		StreamTransformationFilter filter(d);
		filter.Put((const byte*)cipher.data(), cipher.size());
		filter.MessageEnd();

		const size_t ret = filter.MaxRetrievable();
		recovered.resize(ret);
		filter.Get((byte*)recovered.data(), recovered.size());
#endif

	}
	catch(const CryptoPP::Exception& e)
	{
		std::cerr << e.what() << std::endl;
		std::exit(1);
	}

  std::stringstream ss;
  ss.str(recovered);

  std::stringstream errss;


  if (!ifs) {
    errss << "Cannot open file [" << filename << "]" << std::endl;
    if (err) {
      (*err) = errss.str();
    }
    return false;
  }

  std::string baseDir = mtl_basedir ? mtl_basedir : "";
  if (!baseDir.empty()) {
#ifndef _WIN32
    const char dirsep = '/';
#else
    const char dirsep = '\\';
#endif
    if (baseDir[baseDir.length() - 1] != dirsep) baseDir += dirsep;
  }
  MaterialFileReader matFileReader(baseDir);

  return LoadObj(attrib, shapes, materials, warn, err, &ss, &matFileReader,
                 trianglulate, default_vcols_fallback);
}

