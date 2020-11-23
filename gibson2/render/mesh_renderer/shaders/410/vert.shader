#version 410
uniform mat4 V;
uniform mat4 last_V;
uniform mat4 lightV;
uniform mat4 P;
uniform mat4 lightP;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
uniform mat4 last_rot;
uniform mat4 last_trans;
uniform vec3 instance_color;
uniform vec3 diffuse_color;
uniform vec3 uv_transform_param;

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
layout (location=3) in vec3 tangent;
layout (location=4) in vec3 bitangent;

out vec2 theCoords;
out vec3 Normal_world;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Pos_cam_prev;
out vec3 Pos_cam_projected;
out vec3 Diffuse_color;
out mat3 TBN;
out vec4 FragPosLightSpace;
out vec2 Optical_flow;


void main() {
    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);
    vec4 Position_prev = P * last_V * last_trans * last_rot * vec4(position, 1);
    Optical_flow = gl_Position.xy/gl_Position.w - Position_prev.xy/Position_prev.w;
    vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
    FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
    Normal_world = normalize(mat3(pose_rot) * normal); // in world coordinate
    Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate
    vec4 pos_cam4_projected = P * V * pose_trans * pose_rot * vec4(position, 1);
    Pos_cam_projected = pos_cam4_projected.xyz / pos_cam4_projected.w;
    vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
    vec4 pos_cam_prev4 = last_V * last_trans * last_rot * vec4(position, 1);

    Pos_cam = pos_cam4.xyz / pos_cam4.w;
    Pos_cam_prev = pos_cam_prev4.xyz / pos_cam_prev4.w;

    theCoords.x = (cos(uv_transform_param[2]) * texCoords.x * uv_transform_param[0])
                   - (sin(uv_transform_param[2]) * texCoords.y * uv_transform_param[1]);
    theCoords.y = (sin(uv_transform_param[2]) * texCoords.x * uv_transform_param[0])
                   + (cos(uv_transform_param[2]) * texCoords.y * uv_transform_param[1]);
    Instance_color = instance_color;
    Diffuse_color = diffuse_color;
    vec3 T = normalize(vec3(pose_trans * pose_rot * vec4(tangent,   0.0)));
    vec3 B = normalize(vec3(pose_trans * pose_rot * vec4(bitangent, 0.0)));
    vec3 N = normalize(vec3(pose_trans * pose_rot * vec4(normal,    0.0)));
    TBN = mat3(T, B, N);
    FragPosLightSpace = lightP * lightV * pose_trans * pose_rot * vec4(position, 1);
}