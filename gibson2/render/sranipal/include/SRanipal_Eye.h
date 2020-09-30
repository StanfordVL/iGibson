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
///////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "SRanipal_EyeData_v1.h"
#include "SRanipal_EyeData_v2.h"
#ifdef SRANIPAL_EXPORTS
#define SR_ANIPAL __declspec(dllexport)
#else
#define SR_ANIPAL __declspec(dllimport)
#endif

using VivesrEyeDataCallback = void(*)(ViveSR::anipal::Eye::EyeData const &data);

using VivesrEyeDataCallback_v2 = void(*)(ViveSR::anipal::Eye::EyeData_v2 const &data);


extern "C" {

	namespace ViveSR {

		namespace anipal {

			namespace Eye {

				const int ANIPAL_TYPE_EYE = 0;

                const int ANIPAL_TYPE_EYE_V2 = 2;

                /** Check HMD device is ViveProEye or not.
                * @return true : ViveProEye, false : other HMD.
                */
                SR_ANIPAL bool IsViveProEye();

				/** Gets data from anipal's Eye module.
				* @param data ViveSR::anipal::Eye::EyeData
				* @return Indicates the resulting ViveSR::Error status of this method.
				*/
				SR_ANIPAL int GetEyeData(EyeData *data);

                /** Gets data from anipal's Eye module.
                * @param data ViveSR::anipal::Eye::EyeData
                * @return Indicates the resulting ViveSR::Error status of this method.
                */
                SR_ANIPAL int GetEyeData_v2(EyeData_v2 *data);

				/** Sets the parameter of anipal's Eye module.
				* @param parameter ViveSR::anipal::Eye::EyeParameter
				* @return Indicates the resulting ViveSR::Error status of this method.
				*/
				SR_ANIPAL int SetEyeParameter(EyeParameter parameter);

				/** Get the EyeParameter from Eye engine.
				* @param parameter ViveSR::anipal::Eye::EyeParameter
				* @return Indicates the resulting ViveSR::Error status of this method.
				*/
				SR_ANIPAL int GetEyeParameter(EyeParameter *parameter);

				/** Launches anipal's Eye Calibration tool (an overlay program).
				* @param callback (Upcoming feature) A callback method invoked at the end of the calibration process.
				* @return Indicates the resulting ViveSR::Error and CalibrationExitCode status of this method.
				*/
				SR_ANIPAL int LaunchEyeCalibration(void* callback);

				/** Indicate if user is use right calibration configuration and in good IPD and HMD position.
				* @param[out] result Indicate is needed or not.
				* @return Indicates the resulting ViveSR::Error status of this method.
				*/
				SR_ANIPAL int IsUserNeedCalibration(bool* need);

                /* Register a callback function to receive eye camera related data when the module has new outputs.
                [in] function pointer of callback
                [out] error code. please refer Error in ViveSR_Enums.h
                */
                SR_ANIPAL int RegisterEyeDataCallback(VivesrEyeDataCallback callback);

                /* Unegister a callback function to stop receiving eye camera related data.
                [in] function pointer of callback
                [out] error code. please refer Error in ViveSR_Enums.h
                */
                SR_ANIPAL int UnregisterEyeDataCallback(VivesrEyeDataCallback callback);

                /* Register a callback function to receive eye camera related data when the module has new outputs.
                [in] function pointer of callback
                [out] error code. please refer Error in ViveSR_Enums.h
                */
                SR_ANIPAL int RegisterEyeDataCallback_v2(VivesrEyeDataCallback_v2 callback);

                /* Unegister a callback function to stop receiving eye camera related data.
                [in] function pointer of callback
                [out] error code. please refer Error in ViveSR_Enums.h
                */
                SR_ANIPAL int UnregisterEyeDataCallback_v2(VivesrEyeDataCallback_v2 callback);

                /**
                One of return code of API LaunchEyeCalibration()
                */
                enum CalibrationExitCode {
                    CALIBRATION_IS_ALREADY_RUNNING = 2001,
                    OPENVR_DASHBOARD_ACTIVATED = 2002,
                    OPENVR_INIT_FAILED = 2003,
                    OPENVR_OVERLAY_ALREADY_EXISTS = 2004,
                    OPENVR_OVERLAY_CREATE_FAILED = 2005,
                    OPENVR_OVERLAY_INTERFACE_INVALID = 2006,
                    OPENVR_QUIT = 2007
                };
			}
		}
	}
}