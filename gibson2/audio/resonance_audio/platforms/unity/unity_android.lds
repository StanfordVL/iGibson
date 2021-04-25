VERS_1.0 {
  # Export symbols required by Unity.
  global:
    # Export JNI symbols.
    SetListenerGain;
    SetListenerStereoSpeakerMode;
    SetRoomProperties;
    SetRt60ValuesAndProxyRoomProperties;

    UnityGetAudioEffectDefinitions;
  # Hide everything else.
  local:
    *;
};
