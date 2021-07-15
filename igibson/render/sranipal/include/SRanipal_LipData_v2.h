#pragma once
///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//                           License Agreement
//                     For Vive Super Reality Library
//
// Copyright (c) 2019,		HTC Corporation
//
// All rights reserved. Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
namespace ViveSR {

	namespace anipal {
		/**
		* Lip module
		*/
		namespace Lip {

            const int blend_shape_nums = 60;

			/** @enum LipShape
			An enum type for defined muscle.
			*/
            namespace Version2 {
                /**
                * To avoid naming conflict
                */
                enum LipShape_v2
                {
                    None = -1,
                    Jaw_Right = 0,
                    Jaw_Left = 1,
                    Jaw_Forward = 2,
                    Jaw_Open = 3,
                    Mouth_Ape_Shape = 4,
                    Mouth_Upper_Right = 5,
                    Mouth_Upper_Left = 6,
                    Mouth_Lower_Right = 7,
                    Mouth_Lower_Left = 8,
                    Mouth_Upper_Overturn = 9,
                    Mouth_Lower_Overturn = 10,
                    Mouth_Pout = 11,
                    Mouth_Smile_Right = 12,
                    Mouth_Smile_Left = 13,
                    Mouth_Sad_Right = 14,
                    Mouth_Sad_Left = 15,
                    Cheek_Puff_Right = 16,
                    Cheek_Puff_Left = 17,
                    Cheek_Suck = 18,
                    Mouth_Upper_UpRight = 19,
                    Mouth_Upper_UpLeft = 20,
                    Mouth_Lower_DownRight = 21,
                    Mouth_Lower_DownLeft = 22,
                    Mouth_Upper_Inside = 23,
                    Mouth_Lower_Inside = 24,
                    Mouth_Lower_Overlay = 25,
                    Tongue_LongStep1 = 26,
                    Tongue_LongStep2 = 32,
                    Tongue_Down = 30,
					Tongue_Up = 29,
					Tongue_Right = 28,
					Tongue_Left = 27,
					Tongue_Roll = 31,
					Tongue_UpLeft_Morph = 34,
					Tongue_UpRight_Morph = 33,
					Tongue_DownLeft_Morph = 36,
					Tongue_DownRight_Morph = 35,
                    Max = 37,
                };
            }

			/**
			* @struct PredictionData
			* A struct containing all data listed below.
			*/
			struct PredictionData_v2
			{
				float blend_shape_weight[blend_shape_nums];	/*!<The prediction result listing all of @ref LipShape in [0,1].*/
			};

			/**
			* @struct LipData
			* A struct containing all data listed below.
			*/
			struct LipData_v2
			{
				int frame_sequence;				/*!<The frame sequence.*/
				int timestamp;					/*!<The time when the frame was capturing. in millisecond.*/
				char *image;					/*!<The raw buffer. width=800, height=400, channel=1*/
                PredictionData_v2 prediction_data;	/*!<The prediction result listing all of @ref LipShape in [0,1].*/
			};
		}
	}
}