#version 450
uniform mat4 V;
uniform mat4 P;
uniform mat4 pose_rot;
uniform mat4 pose_trans;
uniform vec3 instance_color;
uniform vec3 diffuse_color;

layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 texCoords;
out vec2 theCoords;
out vec3 Normal;
out vec3 FragPos;
out vec3 Normal_cam;
out vec3 Instance_color;
out vec3 Pos_cam;
out vec3 Diffuse_color;

float polyval(float[8] p, float x) {
    float sum = 0;
    for (int i = 0; i < 8; i ++) {
        sum = sum * x + p[i];
    }
    return sum;
}

void main() {

    gl_Position = P * V * pose_trans * pose_rot * vec4(position, 1);

    vec4 tmp_Position = V * pose_trans * pose_rot * vec4(position, 1);

    float zfar = 30;
    float znear = 0.001;

    vec4 tmp_Position2 = tmp_Position;
    tmp_Position2.z -= 1;
    float l = length(tmp_Position2.xyz);

    float param[8] = float[8](0.016647724171699, 0.111071450095377, 0.282341042089142, 0.468925228611114,
        0.774069098188430, 0.861099503685043, 2.274947125380967, 3.018496880968620);
    for (int i = 0; i < 8; i++) param[i] = param[i] * 1e2;

    float xc = 3.100107624460823e+02;
    float yc = 3.099343349150448e+02;
    float c = 1.001332986588265;
    float d = 8.897975970967732e-04;
    float e = 0.001860266075388;

    float lxy = length(tmp_Position.xy);
    if (lxy < 1e-5) lxy = 1e-5;

    float theta = atan(tmp_Position.z / lxy);

    if (theta < 1) {

        float rho = polyval(param, theta);
        float xx = tmp_Position.x / lxy * rho;
        float yy = tmp_Position.y / lxy * rho;
        gl_Position.x = (xx * c + yy * d + xc)/620.0 * 2 - 1;
        gl_Position.y = (xx * e + yy  + yc)/620.0 * 2 - 1;
        gl_Position.z = (l-znear) / (zfar - znear);
        gl_Position.w = 1;
    } else {
        gl_Position = vec4(0,0,1000000,1);
    }

    vec4 world_position4 = pose_trans * pose_rot * vec4(position, 1);
    FragPos = vec3(world_position4.xyz / world_position4.w); // in world coordinate
    Normal = normalize(mat3(pose_rot) * normal); // in world coordinate
    Normal_cam = normalize(mat3(V) * mat3(pose_rot) * normal); // in camera coordinate

    vec4 pos_cam4 = V * pose_trans * pose_rot * vec4(position, 1);
    Pos_cam = pos_cam4.xyz / pos_cam4.w;

    theCoords = texCoords;
    Instance_color = instance_color;
    Diffuse_color = diffuse_color;
}