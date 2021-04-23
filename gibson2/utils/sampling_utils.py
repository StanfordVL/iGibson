from collections import defaultdict

import gibson2
import numpy as np
import pybullet as p
from scipy.stats import truncnorm

from gibson2 import object_states
from gibson2.objects.visual_marker import VisualMarker

_DEFAULT_AABB_OFFSET = 0.1
_DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 0.2
_DEFAULT_MAX_ANGLE_WITH_Z_AXIS = 3 * np.pi / 4
_DEFAULT_MAX_SAMPLING_ATTEMPTS = 10


def get_parallel_rays(source, destination, offset):
    """Given a ray described by a source and a destination, sample 4 parallel rays and return together with input ray.

    The parallel rays start at the corners of a square of edge length `offset` centered on `source`, with the square
    orthogonal to the ray direction. That is, the cast rays are the height edges of a square-base cuboid with bases
    centered on `source` and `destination`.

    :param source: Source of the ray to sample parallel rays of.
    :param destination: Source of the ray to sample parallel rays of.
    :param offset: Orthogonal distance of parallel rays from input ray.
    :return Tuple[List, List] containing sources and destinations of original ray and the parallel rays.
    """
    ray_direction = destination - source

    # Get an orthogonal vector using a random vector.
    random_vector = np.random.rand(3)
    orthogonal_vector_1 = np.cross(ray_direction, random_vector)
    orthogonal_vector_1 /= np.linalg.norm(orthogonal_vector_1)

    # Get a second vector orthogonal to both the ray and the first vector.
    orthogonal_vector_2 = np.cross(ray_direction, orthogonal_vector_1)
    orthogonal_vector_2 /= np.linalg.norm(orthogonal_vector_2)

    orthogonal_vectors = np.array([orthogonal_vector_1, orthogonal_vector_2])
    assert np.all(np.isfinite(orthogonal_vectors))

    # Use the orthogonal vectors to generate some parallel rays.
    ray_offsets = np.array([(-1, -1), (-1, 1), (1, 1), (1, -1)]) * offset
    sources = [source] + [source + np.dot(offsets, orthogonal_vectors) for offsets in ray_offsets]
    destinations = [destination] + [destination + np.dot(offsets, orthogonal_vectors) for offsets in ray_offsets]

    return sources, destinations


def sample_origin_positions(mins, maxes, count, bimodal_mean_fraction, bimodal_stdev_fraction, axis_probabilities):
    """
    Sample ray casting origin positions with a given distribution.

    The way the sampling works is that for each particle, it will sample two coordinates uniformly and one
    using a symmetric, bimodal truncated normal distribution. This way, the particles will mostly be close to the faces
    of the AABB (given a correctly parameterized bimodal truncated normal) and will be spread across each face,
    but there will still be a small number of particles spawned inside the object if it has an interior.

    :param mins: Array of shape (3, ), the minimum coordinate along each axis.
    :param maxes: Array of shape (3, ), the maximum coordinate along each axis.
    :param count: int, Number of origins to sample.
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :return: List of (ray cast axis index, bool whether the axis was sampled from the top side, [x, y, z]) tuples.
    """
    assert len(mins.shape) == 1
    assert mins.shape == maxes.shape

    results = []
    for i in range(count):
        # Get the uniform sample first.
        position = np.random.rand(3)

        # Sample the bimodal normal.
        bottom = (0 - bimodal_mean_fraction) / bimodal_stdev_fraction
        top = (1 - bimodal_mean_fraction) / bimodal_stdev_fraction
        bimodal_sample = truncnorm.rvs(bottom, top, loc=bimodal_mean_fraction, scale=bimodal_stdev_fraction)

        # Pick which axis the bimodal normal sample should go to.
        bimodal_axis = np.random.choice([0, 1, 2], p=axis_probabilities)

        # Choose which side of the axis to sample from. We only sample from the top for the Z axis.
        if bimodal_axis == 2:
            bimodal_axis_top_side = True
        else:
            bimodal_axis_top_side = np.random.choice([True, False])

        # Move sample based on chosen side.
        position[bimodal_axis] = bimodal_sample if bimodal_axis_top_side else 1 - bimodal_sample

        # Scale the position from the standard normal range to the min-max range.
        scaled_position = mins + (maxes - mins) * position

        # Save the result.
        results.append((bimodal_axis, bimodal_axis_top_side, scaled_position))

    return results


def sample_points_on_object(obj,
                            num_points_to_sample,
                            parallel_ray_offset_distance,
                            bimodal_mean_fraction,
                            bimodal_stdev_fraction,
                            axis_probabilities,
                            aabb_offset=_DEFAULT_AABB_OFFSET,
                            max_sampling_attempts=_DEFAULT_MAX_SAMPLING_ATTEMPTS,
                            max_angle_with_z_axis=_DEFAULT_MAX_ANGLE_WITH_Z_AXIS,
                            parallel_ray_normal_angle_tolerance=_DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
                            refuse_downwards=False):
    """
    Samples points on an object's surface using ray casting.

    :param obj: The object to sample points on.
    :param num_points_to_sample: int, the number of points to try to sample.
    :param parallel_ray_offset_distance: float or List[float], the distance of each parallel ray to cast. Can be used to
        find surfaces that are large enough to position known-size objects on. If list, the length should equal
        num_points to sample. Then each i'th sampled position will be sampled using the i'th offset distance (can be
        used to find surfaces to fit differently-sized objects on).
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :param aabb_offset: float, padding for AABB to make sure rays start outside the actual object.
    :param max_sampling_attempts: int, how many times sampling will be attempted for each requested point.
    :param max_angle_with_z_axis: float, maximum angle between hit normal and positive Z axis allowed. Can be used to
        disallow downward-facing hits when refuse_downwards=True.
    :param parallel_ray_normal_angle_tolerance: float, maximum angle between parallel ray normals and main hit normal.
        Used to ensure that the parallel rays hit a flat surface.
    :param refuse_downwards: bool, whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.
    :return: List of num_points_to_sample elements where each element is a tuple in the form of
        (hit_position, hit_normal, {refusal_reason: [refusal_details...]}). Hit positions are set to None when no
        successful sampling happens within the max number of attempts. Refusal details are only filled if the
        debug_sampling flag is globally set to True.
    """
    aabb = obj.states[object_states.AABB].get_value()
    aabb_min = np.array(aabb[0])
    aabb_max = np.array(aabb[1])

    sampling_aabb_min = aabb_min - aabb_offset
    sampling_aabb_max = aabb_max + aabb_offset

    body_id = obj.get_body_id()

    results = [(None, None, defaultdict(list)) for _ in range(num_points_to_sample)]

    for i in range(num_points_to_sample):
        # Sample the starting positions in advance.
        samples = sample_origin_positions(sampling_aabb_min, sampling_aabb_max, max_sampling_attempts,
                                          bimodal_mean_fraction, bimodal_stdev_fraction, axis_probabilities)

        refusal_reasons = results[i][2]

        # Try each sampled position in the AABB.
        for axis, is_top, start_pos in samples:
            # Get the ray casting direction - we want to do it parallel to the sample axis.
            ray_direction = np.array([0, 0, 0])
            ray_direction[axis] = 1
            ray_direction *= -1 if is_top else 1

            # We want to extend our ray until it intersects one of the AABB's faces.
            # Start by getting the distances towards the min and max boundaries of the AABB on each axis.
            point_to_min = aabb_min - start_pos
            point_to_max = aabb_max - start_pos

            # Then choose the distance to the point in the correct direction on each axis.
            closer_point_on_each_axis = np.where(ray_direction < 0, point_to_min, point_to_max)

            # For each axis, find how many times the ray direction should be multiplied to reach the AABB's boundary.
            multiple_to_face_on_each_axis = closer_point_on_each_axis / ray_direction

            # Choose the minimum of these multiples, e.g. how many times the ray direction should be multiplied
            # to reach the nearest boundary.
            multiple_to_face = np.min(multiple_to_face_on_each_axis[np.isfinite(multiple_to_face_on_each_axis)])

            # Finally, use the multiple we found to calculate the point on the AABB boundary that we want to cast our
            # ray until.
            point_on_face = start_pos + ray_direction * multiple_to_face

            # Make sure that we did not end up with all NaNs or infinities due to division issues.
            assert not np.any(np.isnan(point_on_face)) and not np.any(np.isinf(point_on_face))

            ray_offset_distance = parallel_ray_offset_distance
            # If we have a list of offset distances, pick the distance for this particular sample we're getting.
            if isinstance(ray_offset_distance, list):
                assert len(ray_offset_distance) == num_points_to_sample, "Need as many offsets as samples requested."
                ray_offset_distance = ray_offset_distance[i]

            sources, destinations = get_parallel_rays(start_pos, point_on_face, ray_offset_distance)

            # Time to cast the rays.
            res = p.rayTestBatch(rayFromPositions=sources, rayToPositions=destinations)

            # Check that the center ray has hit our object.
            if res[0][0] != body_id:
                if gibson2.debug_sampling:
                    refusal_reasons["missed_object"].append("hit %d" % body_id)
                continue

            # Get the candidate position & normal.
            hit_pos = np.array(res[0][3])
            hit_normal = np.array(res[0][4])
            hit_normal /= np.linalg.norm(hit_normal)

            # Reject anything facing more than 45deg downwards if requested.
            if refuse_downwards:
                hit_angle_with_z = np.arccos(np.clip(np.dot(hit_normal, np.array([0, 0, 1])), -1.0, 1.0))
                if hit_angle_with_z > max_angle_with_z_axis:
                    if gibson2.debug_sampling:
                        refusal_reasons["downward_normal"].append("normal %r" % hit_normal)
                    continue

            # Check that all rays hit the object.
            parallel_hit_body_ids = [ray_res[0] for ray_res in res[1:]]
            if not all(hit_body_id == body_id for hit_body_id in parallel_hit_body_ids):
                if gibson2.debug_sampling:
                    refusal_reasons["parallel_hit_missed"].append("normal %r" % parallel_hit_body_ids)
                continue

            # Check that none of the parallel rays' hit normal differs from center ray by more than threshold.
            parallel_hit_positions = np.array([ray_res[3] for ray_res in res[1:]])
            parallel_hit_normals = np.array([ray_res[4] for ray_res in res[1:]])
            parallel_hit_normals /= np.linalg.norm(parallel_hit_normals, axis=1)[:, np.newaxis]

            parallel_hit_main_hit_dot_products = np.clip(np.dot(parallel_hit_normals, hit_normal), -1.0, 1.0)
            parallel_hit_normal_angles_to_hit_normal = np.arccos(parallel_hit_main_hit_dot_products)

            all_rays_hit_with_similar_normal = np.all(
                parallel_hit_normal_angles_to_hit_normal < parallel_ray_normal_angle_tolerance)
            if not all_rays_hit_with_similar_normal:
                if gibson2.debug_sampling:
                    refusal_reasons["parallel_hit_angle_off"].append(
                        "normal %r, hit normals %r, hit angles %r" % (
                            hit_normal, parallel_hit_normals, parallel_hit_normal_angles_to_hit_normal))
                continue

            # Debug markers
            # TODO (Cem): Either make this possible to toggle on/off or delete it.
            # Kept for now for debugging cases that seem to spring up often.
            # from gibson2 import simulator
            # color = np.concatenate([np.random.rand(3), [1]])
            # for vec in parallel_hit_positions:
            #     simulator.SIM.import_object(VisualMarker(
            #         rgba_color=color,
            #         radius=0.01,
            #         initial_offset=vec
            #     ))

            # Compute a rotation from the default AABB to the sampled position.
            center_to_bottom_left_on_original = np.array([-1, -1]) * parallel_ray_offset_distance
            center_to_bottom_left_on_original = np.concatenate([center_to_bottom_left_on_original, [0]])
            center_to_bottom_left_on_original /= np.linalg.norm(center_to_bottom_left_on_original)

            center_to_bottom_left_on_cast = parallel_hit_positions[0] - hit_pos
            center_to_bottom_left_on_cast /= np.linalg.norm(center_to_bottom_left_on_cast)

            dot = np.dot(center_to_bottom_left_on_original, center_to_bottom_left_on_cast)
            if dot > 0.999:
                # There was practically no rotation, so use the identity quaternion
                rotation = np.array([0, 0, 0, 1])
            else:
                # Compute the rotation quaternion
                axis = np.cross(center_to_bottom_left_on_original, center_to_bottom_left_on_cast)
                rotation = np.concatenate([axis, [1 + dot]])
                rotation /= np.linalg.norm(rotation)

            # We've found a nice attachment point. Let's go.
            results[i] = (hit_pos, hit_normal, rotation, refusal_reasons)
            break

    return results
