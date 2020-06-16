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
			/** @enum LipShape
			An enum type for defined muscle.
			*/
			enum LipShape
			{
				Jaw_Forward = 0,
				Jaw_Right = 1,
				Jaw_Left = 2,
				Jaw_Open = 3,
				Mouth_Ape_Shape = 4,
				Mouth_O_Shape = 5,
				Mouth_Pout = 6,
				Mouth_Lower_Right = 7,
				Mouth_Lower_Left = 8,
				Mouth_Smile_Right = 9,
				Mouth_Smile_Left = 10,
				Mouth_Sad_Right = 11,
				Mouth_Sad_Left = 12,
				Cheek_Puff_Right = 13,
				Cheek_Puff_Left = 14,
				Mouth_Lower_Inside = 15,
				Mouth_Upper_Inside = 16,
				Mouth_Lower_Overlay = 17,
				Mouth_Upper_Overlay = 18,
				Cheek_Suck = 19,
				Mouth_LowerRight_Down = 20,
				Mouth_LowerLeft_Down = 21,
				Mouth_UpperRight_Up = 22,
				Mouth_UpperLeft_Up = 23,
				Mouth_Philtrum_Right = 24,
				Mouth_Philtrum_Left = 25,
				None = 26,
			};

			/**
			* @struct PredictionData
			* A struct containing all data listed below.
			*/
			struct PredictionData
			{
				float blend_shape_weight[27];	/*!<The prediction result listing all of @ref LipShape in [0,1].*/
			};

			/**
			* @struct LipData
			* A struct containing all data listed below.
			*/
			struct LipData
			{
				int frame_sequence;				/*!<The frame sequence.*/
				int timestamp;					/*!<The time when the frame was capturing. in millisecond.*/
				char *image;					/*!<The raw buffer. width=800, height=400, channel=1*/
				PredictionData prediction_data;	/*!<The prediction result listing all of @ref LipShape in [0,1].*/
			};
		}
	}
}