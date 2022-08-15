#version 460
// plane in front of the eye
#define TEXTURE_SCALE 2.0
#define TEXTURE_DISTANCE 1000.0

// This model was obtained via machine learning, using a
// particle swarm optimisation algorithm. The cost gives a
// rough estimate for the quality of the model (the lower
// the cost, the better).
// Cost:  0.11083103563252572
#define N_AIR 1.0
#define N_CORNEA 1.64099928591927
#define N_ANTERIOR_CHAMBER 1.4532300393624265
#define N_LENS 1.4393602250536033
#define N_VITREOUS_HUMOUR 1.362681339093861

#define OUTER_CORNEA_RADIUS 7.873471062713062
#define INNER_CORNEA_RADIUS 4.7658074271207695
#define VITREOUS_HUMOUR_RADIUS 11.704040541034434

#define OUTER_CORNEA_CENTER vec3(0., 0., 6.673155500500702)
#define INNER_CORNEA_CENTER vec3(0., 0., 7.877094062114152)
#define VITREOUS_HUMOUR_CENTER vec3(0., 0., 0.0)

#define OUTER_LENS_RADIUS_A 5.0719602142230795E-11
#define OUTER_LENS_RADIUS_B -4.728025830060936E-6
#define OUTER_LENS_RADIUS_C 8.768369917723753
#define INNER_LENS_RADIUS_A 1.8872104415414247E-11
#define INNER_LENS_RADIUS_B 1.3008349405807643E-5
#define INNER_LENS_RADIUS_C 7.237406273489772

#define OUTER_LENS_TIP_A -3.867155451878399E-11
#define OUTER_LENS_TIP_B 1.5465450056679198E-7
#define OUTER_LENS_TIP_C 8.839055805878198
#define INNER_LENS_TIP_A -8.525281533727847E-12
#define INNER_LENS_TIP_B 3.777299048943033E-6
#define INNER_LENS_TIP_C 7.03207918882336
// end of model

#define CORNEA_MAP_FACTOR 0.2

int u_samplecount = 4;
float u_depth_min = 100.0;
float u_depth_max = 1800.0;
float u_dir_calc_scale = 1.0;
float u_astigmatism_ecc_mm = 0.0;
float u_astigmatism_angle_deg = 0.0;
vec2 u_lens_position = vec2(0.0, 0.0);
float u_eye_distance_center = 0.0;

uniform float u_near_point = 0.0;
uniform float u_far_point = 1000000000.0;
uniform float u_near_vision_factor = 0.0;
uniform float u_far_vision_factor = 0.0;
uniform int u_active = 0;

uniform sampler2D s_color;
uniform sampler2D s_depth;

in vec2 TexCoords;
out vec4 FragColor;


struct Simulation
{
  vec4 color;
};


// alters the direction of a ray according to the imperfection in the cornea specified by the cornea map
// vec3 applyCorneaImperfection(vec3 pos, vec3 dir) {
//     // scale 3D ray to 2D texture coordinates for lookup
//     // The size of the cornea (~5.5mm radius) is mapped to [0;1],[0;1] texture coordinates
//     vec2 corneaMapSample = texture(s_cornea, pos.xy * vec2(0.17, -0.17) + 0.5).rg;
//     vec3 deviation = vec3((corneaMapSample - 0.5) * vec2(CORNEA_MAP_FACTOR), 0.0);
//     return dir + deviation;
// }

// calculates the radius of curvature of the outer lens surface
// while taking accomodation into account.
// focalLength: the distance between the frontmost point of the
//              vitreous humour and the focused point
float getOuterLensRadius(float focalLength) {
    return OUTER_LENS_RADIUS_A * focalLength * focalLength
        + OUTER_LENS_RADIUS_B * focalLength
        + OUTER_LENS_RADIUS_C;
}

// calculates the center of the sphere representing the outer
// lens surface while taking accomodation into account.
// focalLength: the distance between the frontmost point of the
//              vitreous humour and the focused point
vec3 getOuterLensCenter(float focalLength) {
    float outerLensTip = OUTER_LENS_TIP_A * focalLength * focalLength
        + OUTER_LENS_TIP_B * focalLength
        + OUTER_LENS_TIP_C;
    return vec3(0.0, 0.0, outerLensTip - getOuterLensRadius(focalLength));
}

// calculates the radius of curvature of the inner lens surface
// while taking accomodation into account.
// focalLength: the distance between the frontmost point of the
//              vitreous humour and the focused point
float getInnerLensRadius(float focalLength) {
    return INNER_LENS_RADIUS_A * focalLength * focalLength
        + INNER_LENS_RADIUS_B * focalLength
        + INNER_LENS_RADIUS_C;
}

// calculates the center of the sphere representing the inner
// lens surface while taking accomodation into account.
// focalLength: the distance between the frontmost point of the
//              vitreous humour and the focused point
vec3 getInnerLensCenter(float focalLength) {
    float innerLensTip = INNER_LENS_TIP_A * focalLength * focalLength
        + INNER_LENS_TIP_B * focalLength
        + INNER_LENS_TIP_C;
    return vec3(0.0, 0.0, innerLensTip + getInnerLensRadius(focalLength));
}

float getNAnteriorChamber(float nAnteriorChamberFactor) {
    return N_ANTERIOR_CHAMBER * nAnteriorChamberFactor;
}

float getDepth(vec2 pos) {
    float ndc = texture(s_depth, pos).r * 2.0 - 1.0; 
    return (2.0 * u_depth_min * u_depth_max) / (u_depth_max + u_depth_min - ndc * (u_depth_max - u_depth_min));
    // return u_depth_max - texture(s_depth, pos).r * (u_depth_max - u_depth_min);
}

vec3 getTargetLocation(in vec3 start, in vec3 dir){
    float t = (getDepth(TexCoords + (u_lens_position / (vec2(1920, 1080) / 2))) + VITREOUS_HUMOUR_RADIUS - start.z) / dir.z;
    start += dir * t;
    return start;
}

// intesects the ray with the image and returns the color of the texture at this position
vec4 getColorWithRay(in vec3 target) {
    return texture(s_color, target.xy / ((target.z - VITREOUS_HUMOUR_RADIUS) * TEXTURE_SCALE) + 0.5);
}

void getRay(inout vec3 start, inout vec3 dir){
	// TODO subtract (0.5, 0.5) ?
	start = vec3((TexCoords.xy - 0.5) * 2000.0, 1000.0 + VITREOUS_HUMOUR_RADIUS);
	dir = normalize(vec3(0.0, 0.0, VITREOUS_HUMOUR_RADIUS) - start);
}

//rPos = ray position
//rDir = ray direction
//cPos = sphere position
//cRad = sphere radius
// returns a point where the ray intersects the sphere
vec3 rayIntersectSphere(vec3 rPos, vec3 rDir, vec3 cPos, float cRad) {
    float a = dot(rDir, rDir);
    float b = 2.0 * dot(rDir, (rPos - cPos));
    float c = dot(rPos - cPos, rPos - cPos) - cRad * cRad;
    float disk = sqrt(b * b - 4.0 * a * c);
    float t1 = (-b + disk) / (2.0 * a);
    float t2 = (-b - disk) / (2.0 * a);

    if (t1 < t2) {
        return (t1 >= 0.0 ? t1 : t2) * rDir + rPos;
    } else {
        return (t2 >= 0.0 ? t2 : t1) * rDir + rPos;
    }
}


struct RefractInfo{
    vec3 position;
    vec3 normal;
};

RefractInfo rayIntersectEllipsoid(vec3 rPos, vec3 rDir, vec3 cPos, float cRad) {
    float mul = radians(u_astigmatism_angle_deg);
    // float mul = 0.25 * 0;

    // We start off by defining the three main axis of the ellispoid.
    vec3 a = vec3(cRad -0.094, 0,0); //experimental value, remove when cubemap works
    vec3 b = vec3(0     , cRad - 0.078 - u_astigmatism_ecc_mm ,    0); //experimental value, remove when cubemap works
    vec3 c = vec3(0     , 0         , cRad);

    // Now we allow for rotation along the Z axis (viewing direction)
    // This enables us to simulate Atigmatism obliquus, with the main axis not in line with world-coordinates

    // The matrix for rotating the cornea elipsoid
    mat3 rot = mat3(
        cos(mul), sin(mul), 0,
        -sin(mul), cos(mul) , 0,
               0,         0, 1
    );

    // rotate the ellispoid (main axis)
    a = rot * a;
    b = rot * b;
    c = rot * c;

    // Now that the ellipsoid is defined, we prepare to transform the whole space, so that the ellipsoid
    // becomes a up-oriented-unit-sphere (|a|=|b|=|c| ; a = a.x, r=1), and center in the origin.
    // inspired by http://www.illusioncatalyst.com/notes_files/mathematics/line_nu_sphere_intersection.php



    mat4 T = mat4(
        1.0, 0, 0, 0,
        0, 1.0, 0, 0,
        0, 0, 1.0, 0,
        cPos.x, cPos.y, cPos.z, 1.0
    );

    vec3 an = normalize(a);
    vec3 bn = normalize(b);
    vec3 cn = normalize(c);

    mat4 R = mat4(
        an.x, an.y, an.z, 0,
        bn.x, bn.y, bn.z, 0,
        cn.x, cn.y, cn.z, 0,
        0   , 0   , 0   , 1
    );

    mat4 S = mat4(
        length(a), 0, 0, 0,
        0, length(b), 0, 0,
        0, 0, length(c), 0,
        0, 0, 0, 1.0
    );

    // Transform the Ray according to the inverse transformation, that once created the ellipsoid from the sphere
    // We do **not** translate the direction component of the ray, since it would change the direction of the ray and hence the resulting normal
    vec4 srPos =  inverse(S) * inverse(T) * inverse(R) * vec4(rPos,1.0f) ;
    vec4 srDir =  inverse(S) * inverse(R) * vec4(rDir,1.0f);
        
    // Since we transformed everything, sphere position is now the origin and its radius is 1
    vec3 sTarget = rayIntersectSphere(srPos.xyz, srDir.xyz, vec3(0.0), 1.0);
    
    // We can gradually transform everything back to its original state.
    // The first step is scaling the space to once again get the original ellipsoid shape
    // This is used for calculating the normal. However, the normal caluclation assumes a,b,c, to be aligned with
    //   the coordinate axis, so apply the rotation to the normal instead
    vec4 rel_target = S * vec4(sTarget,1.0) ;

    // to calculate the normal, we use n = x_0/a^2 , y_0/b^2, z_0/c^2 with the ellipsoid centered at the origin
    vec4 normal = vec4(rel_target.x/pow(length(a),2), rel_target.y/pow(length(b),2), rel_target.z/pow(length(c),2), 1.0);
    // restore the original ellipsoid rotation
    normal = R * normal;

    // the intersection point of the ray with the ellipsoid. All transformations applied initially to the ray-position
    // are now applied as inverted matrices (so the original ones, that created the ellipsoid) and in inverse order.
    vec3 world_target = (R*(T*(S * vec4(sTarget, 1.0)))).rgb;


    return RefractInfo(
        world_target.xyz,
        normal.xyz
    );
}

// sends the ray through all four surfaces in the optical system and returns the color, position, error and uncertainty
// of the texture at the intersection point.
Simulation getColorSample(vec3 start, vec2 aim, float focalLength, float nAnteriorChamberFactor) {
    vec3 dir = normalize(vec3(aim, VITREOUS_HUMOUR_RADIUS) - start);

    start = rayIntersectSphere(start, dir, getInnerLensCenter(focalLength), getInnerLensRadius(focalLength));
    dir = refract(dir, normalize(start - getInnerLensCenter(focalLength)), N_VITREOUS_HUMOUR / N_LENS);

    start = rayIntersectSphere(start, dir, getOuterLensCenter(focalLength), getOuterLensRadius(focalLength));
    dir = refract(dir, -normalize(start - getOuterLensCenter(focalLength)), N_LENS / getNAnteriorChamber(nAnteriorChamberFactor));

    start = rayIntersectSphere(start, dir, INNER_CORNEA_CENTER, INNER_CORNEA_RADIUS);
    dir = refract(dir, -normalize(start - INNER_CORNEA_CENTER), getNAnteriorChamber(nAnteriorChamberFactor) / N_CORNEA);

    RefractInfo ri = rayIntersectEllipsoid(start, dir, OUTER_CORNEA_CENTER, OUTER_CORNEA_RADIUS);
    start = ri.position;

    // we modeled the cornea as an ellipsoid instead of a sphere.
    // although the lens can also contribute to a asigmatism, the result is the same, as if the cornea would introduce all the error 
    // start = rayIntersectSphere(start, dir, OUTER_CORNEA_CENTER, OUTER_CORNEA_RADIUS);


    dir = refract(dir, -normalize(ri.normal), N_CORNEA / N_AIR);
    // dir = refract(dir, -normalize(start - OUTER_CORNEA_CENTER), N_CORNEA / N_AIR);
    // dir = applyCorneaImperfection(start, dir);

    start += vec3(u_lens_position, 0) + vec3(u_eye_distance_center,0,0);

    // position where the ray hits the image, including depth
    vec3 target = getTargetLocation(start, dir);

    // The color and the error values are sampled from the original textures
    vec4 color = getColorWithRay(target);
    Simulation sim = Simulation(
        color
    );
    return sim;
}


void main() {
    if (1 == u_active) {
        // get normal
        vec3 start_normal = vec3(0.0, 0.0, 0.0), dir = vec3(0.0, 0.0, 0.0);
        getRay(start_normal, dir);
        
        float distance = length(vec3((TexCoords.xy - 0.5) * 2000.0, 1000.0));

        start_normal = rayIntersectSphere(start_normal, dir, OUTER_CORNEA_CENTER, OUTER_CORNEA_RADIUS);
        dir = refract(dir, normalize(start_normal - OUTER_CORNEA_CENTER), N_AIR/N_CORNEA);

        start_normal = rayIntersectSphere(start_normal, dir, INNER_CORNEA_CENTER, INNER_CORNEA_RADIUS);
        dir = refract(dir, normalize(start_normal - INNER_CORNEA_CENTER), N_CORNEA/N_ANTERIOR_CHAMBER);

        start_normal = rayIntersectSphere(start_normal, dir, getOuterLensCenter(distance), getOuterLensRadius(distance));
        dir = refract(dir, normalize(start_normal - getOuterLensCenter(distance)), N_ANTERIOR_CHAMBER/N_LENS);

        start_normal = rayIntersectSphere(start_normal, dir, getInnerLensCenter(distance), getInnerLensRadius(distance));
        dir = refract(dir, -normalize(start_normal - getInnerLensCenter(distance)), N_LENS/N_VITREOUS_HUMOUR);


        start_normal = rayIntersectSphere(start_normal, dir, VITREOUS_HUMOUR_CENTER, VITREOUS_HUMOUR_RADIUS);

        start_normal = normalize(start_normal) / 2. + 0.5;

        // Start of lens simulation
        FragColor = vec4(0.5);
        vec3 start = (start_normal * 2.0 - 1.0) * VITREOUS_HUMOUR_RADIUS;
        vec3 original_color = texture(s_color, TexCoords).rgb;

        float sampleCount = 0.0;
        vec4 color = vec4(0.0);
        vec2 avg_target = vec2(0.0);

   
        // prepare accomodation
        // This implementation focuses on all possible depths: for each pixel, the lens can focus on the individual depth
        float focalLength = length(vec3(
            (TexCoords.xy - 0.5) * TEXTURE_SCALE,
            getDepth(TexCoords.xy)));        
       
        // This implementation focuses soley on the depth that is in the middle of the screen
        // float focalLength = length(
        //     vec3(
        //         (TexCoords.xy - 0.5) * TEXTURE_SCALE,
        //         u_depth_max - texture(s_depth, vec2(0.5,0.5)).r * (u_depth_max - u_depth_min)
        //     )
        // );  


        float nAnteriorChamberFactor = 1.0;
        if (focalLength > u_far_point) {
            // TODO factor 0.08
            nAnteriorChamberFactor = 1.0 / pow((focalLength - u_far_point) / focalLength + 1.0, 0.08 * u_far_vision_factor);
            focalLength = u_far_point;
        } else if (focalLength < u_near_point) {
            // TODO factor 0.12
            nAnteriorChamberFactor = pow(focalLength / u_near_point, 0.12 * u_near_vision_factor);
            focalLength = u_near_point;
        }

        Simulation rays[17];

        // The higher the sampleCount, the more rays are cast. Rays are cast in this fashion,
        // where the number indicates the minimum sampleCount required for this ray to be cast:
        // 3   2   3
        //   4 4 4
        // 2 4 1 4 2
        //   4 4 4
        // 3   2   3
        // Since the ray direction is only altered a small amount compared to the radius, 
        //     it is safe to assume that all of them will still hit the lens.
        switch (u_samplecount) {
        case 4:
            sampleCount += 8.0;
            rays[0] = getColorSample(start, vec2(-0.5, -0.5), focalLength, nAnteriorChamberFactor);
            rays[1] = getColorSample(start, vec2(-0.5, 0.5), focalLength, nAnteriorChamberFactor);
            rays[2] = getColorSample(start, vec2(0.5, -0.5), focalLength, nAnteriorChamberFactor);
            rays[3] = getColorSample(start, vec2(0.5, 0.5), focalLength, nAnteriorChamberFactor);
            rays[4] = getColorSample(start, vec2(-0.5, 0.0), focalLength, nAnteriorChamberFactor);
            rays[5] = getColorSample(start, vec2(0.5, 0.0), focalLength, nAnteriorChamberFactor);
            rays[6] = getColorSample(start, vec2(0.0, -0.5), focalLength, nAnteriorChamberFactor);
            rays[7] = getColorSample(start, vec2(0.0, 0.5), focalLength, nAnteriorChamberFactor);
        case 3:
            sampleCount += 4.0;
            rays[8] = getColorSample(start, vec2(-1.0, -1.0), focalLength, nAnteriorChamberFactor);
            rays[9] = getColorSample(start, vec2(-1.0, 1.0), focalLength, nAnteriorChamberFactor);
            rays[10] = getColorSample(start, vec2(1.0, -1.0), focalLength, nAnteriorChamberFactor);
            rays[11] = getColorSample(start, vec2(1.0, 1.0), focalLength, nAnteriorChamberFactor);
        case 2:
            sampleCount += 4.0;
            rays[12] = getColorSample(start, vec2(-1.0, 0.0), focalLength, nAnteriorChamberFactor);
            rays[13] = getColorSample(start, vec2(1.0, 0.0), focalLength, nAnteriorChamberFactor);
            rays[14] = getColorSample(start, vec2(0.0, -1.0), focalLength, nAnteriorChamberFactor);
            rays[15] = getColorSample(start, vec2(0.0, 1.0), focalLength, nAnteriorChamberFactor);
        case 1:
            sampleCount += 1.0;
            rays[16] = getColorSample(start, vec2(0.0, 0.0), focalLength, nAnteriorChamberFactor);
        }

        // calculate the mean color and target vector
        // Thie mean target vector is the "center" of the scatter
        for (int i = 0; i < 17; i++)
            color += rays[i].color;
        color /= sampleCount;

        FragColor = color;


    } else {
        FragColor = texture(s_color, TexCoords);
    }

}