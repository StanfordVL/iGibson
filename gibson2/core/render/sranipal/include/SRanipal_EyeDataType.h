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
#include <stdint.h>

namespace ViveSR {
    namespace anipal {
        namespace Eye {
#pragma region basic_math_data_type
            /**
            * @struct Vector3
            * A basic 3d vector type
            */
            struct Vector3
            {
                union
                {
                    struct
                    {
                        float x;
                        float y;
                        float z;
                    };
                    float elem_[3];
                };
            };
            /**
            *  @struct Vector2
            *  A basic 2d vector type
            */
            struct Vector2
            {
                union
                {
                    struct
                    {
                        float x;
                        float y;
                    };
                    float elem_[2];
                };
            };
            /**
            * @struct Matrix4x4
            *  A basic affine matrix in 3D space
            */
            struct Matrix4x4
            {
                union
                {
                    struct
                    {
                        float m00;
                        float m33;
                        float m23;
                        float m13;
                        float m03;
                        float m32;
                        float m22;
                        float m02;
                        float m12;
                        float m21;
                        float m11;
                        float m01;
                        float m30;
                        float m20;
                        float m10;
                        float m31;
                    };
                    float elem_[16];
                };
            };
#pragma endregion
#pragma region bit_mask_enum
            /** @enum SingleEyeDataValidity
            An enum type for getting validity from the structure: eye data's bitmask
            */
            enum SingleEyeDataValidity {
                SINGLE_EYE_DATA_GAZE_ORIGIN_VALIDITY,/*!<The validity of the origin of gaze of the eye data*/
                SINGLE_EYE_DATA_GAZE_DIRECTION_VALIDITY,/*!<The validity of the direction of gaze of the eye data*/
                SINGLE_EYE_DATA_PUPIL_DIAMETER_VALIDITY,/*!<The validity of the diameter of gaze of the eye data*/
                SINGLE_EYE_DATA_EYE_OPENNESS_VALIDITY,/*!<The validity of the openness of the eye data>*/
                SINGLE_EYE_DATA_PUPIL_POSITION_IN_SENSOR_AREA_VALIDITY/*!<The validity of normalized position of pupil*/
            };

            enum TrackingImprovement
            {
                TRACKING_IMPROVEMENT_USER_POSITION_HMD,
                TRACKING_IMPROVEMENT_CALIBRATION_CONTAINS_POOR_DATA,
                TRACKING_IMPROVEMENT_CALIBRATION_DIFFERENT_BRIGHTNESS,
                TRACKING_IMPROVEMENT_IMAGE_QUALITY,
                TRACKING_IMPROVEMENT_INCREASE_EYE_RELIEF,
            };

            /** A utility function for decoding bits by bitmask.
            @param bits the input as bitmask
            @param position the the position of the bits.
            @return The status of a bit.
            @sa VerboseDataValidity, EyeDataValidity
            */
            inline bool DecodeBitMask(const uint64_t& bits, unsigned char position) {
                return bits&(1 << position);
            }
            /** A utility function for encoding bits by bitmask
            @param position the the position of the bits.
            @param status the status you would set.
            @param[out] bits the bits.
            @sa VerboseDataValidity, EyeDataValidity
            */
            inline void EecodeBitMask(unsigned char position, bool status, uint64_t& bits) {
                uint64_t the_mask = (((uint64_t)1) << position);
                if (status)
                    bits |= the_mask;
                else
                    bits &= ~the_mask;
            }
#pragma endregion
#pragma region eye_parameter
            struct TrackingImprovements
            {
                int count;
                union
                {
                    struct
                    {
                        TrackingImprovement t0;
                        TrackingImprovement t1;
                        TrackingImprovement t2;
                        TrackingImprovement t3;
                        TrackingImprovement t4;
                        TrackingImprovement t5;
                        TrackingImprovement t6;
                        TrackingImprovement t7;
                        TrackingImprovement t8;
                        TrackingImprovement t9;
                    };
                    TrackingImprovement elem_[10];
                };
            };

            /** @struct GazeRayParameter
            * A struct containing all data listed below.
            */
            struct GazeRayParameter {
                double sensitive_factor;/*!<The sensitive factor of gaze ray in [0,1]. The bigger factor is, the more sensitive the gaze ray is.*/
            };

            /** @struct EyeParameter
            * A struct containing all data listed below.
            */
            struct EyeParameter {
                GazeRayParameter gaze_ray_parameter;
            };
#pragma endregion
#pragma region VerboseData
            			/** @struct SingleEyeData
			* A struct containing status related an eye.
			*/
			typedef struct SingleEyeData
			{
				uint64_t eye_data_validata_bit_mask;/*!<The bits containing all validity for this frame.*/
				Vector3 gaze_origin_mm;/*!<The point in the eye from which the gaze ray originates in meter miles.(right-handed coordinate system)*/
				Vector3 gaze_direction_normalized;/*!The normalized gaze direction of the eye in [0,1].(right-handed coordinate system)*/
				float pupil_diameter_mm;/*!<The diameter of the pupil in meter miles*/
				float eye_openness;/*!<A value representing how open the eye is.*/
				Vector2 pupil_position_in_sensor_area;/*!<The normalized position of a pupil in [0,1]*/
			}SingleEyeData;

			typedef struct CombinedEyeData
			{
				SingleEyeData eye_data;
				bool convergence_distance_validity;
				float convergence_distance_mm;
			};

			/** @struct VerboseData
			* A struct containing all data listed below.
			*/
			typedef struct VerboseData
			{
				SingleEyeData left;/*!<A instance of the struct as @ref EyeData related to the left eye*/
				SingleEyeData right;/*!<A instance of the struct as @ref EyeData related to the right eye*/
				CombinedEyeData combined;/*!<A instance of the struct as @ref EyeData related to the combined eye*/
				TrackingImprovements tracking_improvements;
			}VerboseData;
#pragma endregion
        }
    }
}