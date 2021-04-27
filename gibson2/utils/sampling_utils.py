import itertools
from collections import defaultdict, Counter

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from scipy.stats import truncnorm

import gibson2
from gibson2 import object_states
from gibson2.objects.visual_marker import VisualMarker

_DEFAULT_AABB_OFFSET = 0.1
_DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE = 0.2
_DEFAULT_MAX_ANGLE_WITH_Z_AXIS = 3 * np.pi / 4
_DEFAULT_MAX_SAMPLING_ATTEMPTS = 10
_DEFAULT_CUBOID_BOTTOM_PADDING = 0.01

# We will cast an additional parallel ray for each additional this much distance.
_DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE = 0.1


def get_parallel_rays(source, destination, offset,
                      new_ray_per_horizontal_distance=_DEFAULT_NEW_RAY_PER_HORIZONTAL_DISTANCE):
    """Given a ray described by a source and a destination, sample parallel rays and return together with input ray.

    The parallel rays start at the corners of a square of edge length `offset` centered on `source`, with the square
    orthogonal to the ray direction. That is, the cast rays are the height edges of a square-base cuboid with bases
    centered on `source` and `destination`.

    :param source: Source of the ray to sample parallel rays of.
    :param destination: Source of the ray to sample parallel rays of.
    :param offset: Orthogonal distance of parallel rays from input ray.
    :param new_ray_per_horizontal_distance: Step in offset beyond which an additional split will be applied in the
        parallel ray grid (which at minimum is 3x3 at the AABB corners & center).
    :return Tuple[List, List, Array[W, H, 3]] containing sources and destinations of original ray and the unflattened,
        untransformed grid in object coordinates.
    """
    ray_direction = destination - source

    # Get an orthogonal vector using a random vector.
    random_vector = np.random.rand(3)
    orthogonal_vector_1 = np.cross(ray_direction, random_vector)
    orthogonal_vector_1 /= np.linalg.norm(orthogonal_vector_1)

    # Get a second vector orthogonal to both the ray and the first vector.
    orthogonal_vector_2 = -np.cross(ray_direction, orthogonal_vector_1)
    orthogonal_vector_2 /= np.linalg.norm(orthogonal_vector_2)

    orthogonal_vectors = np.array([orthogonal_vector_1, orthogonal_vector_2])
    assert np.all(np.isfinite(orthogonal_vectors))

    # Convert the offset into a 2-vector if it already isn't one.
    offset = np.array([1, 1]) * offset

    # Compute the grid of rays
    steps = (offset / new_ray_per_horizontal_distance).astype(int) * 2 + 1
    steps = np.maximum(steps, 3)
    x_range = np.linspace(-offset[0], offset[0], steps[0])
    y_range = np.linspace(-offset[1], offset[1], steps[1])
    ray_grid = np.dstack(np.meshgrid(x_range, y_range, indexing="ij"))
    ray_grid_flattened = ray_grid.reshape(-1, 2)

    # Apply the grid onto the orthogonal vectors to obtain the rays.
    sources = [source + np.dot(offsets, orthogonal_vectors) for offsets in ray_grid_flattened]
    destinations = [destination + np.dot(offsets, orthogonal_vectors) for offsets in ray_grid_flattened]

    return sources, destinations, ray_grid


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


def sample_cuboid_on_object(obj,
                            num_samples,
                            cuboid_dimensions,
                            bimodal_mean_fraction,
                            bimodal_stdev_fraction,
                            axis_probabilities,
                            bottom_padding=_DEFAULT_CUBOID_BOTTOM_PADDING,
                            aabb_offset=_DEFAULT_AABB_OFFSET,
                            max_sampling_attempts=_DEFAULT_MAX_SAMPLING_ATTEMPTS,
                            max_angle_with_z_axis=_DEFAULT_MAX_ANGLE_WITH_Z_AXIS,
                            parallel_ray_normal_angle_tolerance=_DEFAULT_PARALLEL_RAY_NORMAL_ANGLE_TOLERANCE,
                            refuse_downwards=False):
    """
    Samples points on an object's surface using ray casting.

    :param obj: The object to sample points on.
    :param num_samples: int, the number of points to try to sample.
    :param cuboid_dimensions: Float sequence of len 3, the size of the empty cuboid we are trying to sample. Can also
        provice list of cuboid dimension triplets in which case each i'th sample will be sampled using the i'th triplet.
    :param bimodal_mean_fraction: float, the mean of one side of the symmetric bimodal distribution as a fraction of the
        min-max range.
    :param bimodal_stdev_fraction: float, the standard deviation of one side of the symmetric bimodal distribution as a
        fraction of the min-max range.
    :param axis_probabilities: Array of shape (3, ), the probability of ray casting along each axis.
    :param bottom_padding: float, padding to leave between object surface and cuboid
    :param aabb_offset: float, padding for AABB to make sure rays start outside the actual object.
    :param max_sampling_attempts: int, how many times sampling will be attempted for each requested point.
    :param max_angle_with_z_axis: float, maximum angle between hit normal and positive Z axis allowed. Can be used to
        disallow downward-facing hits when refuse_downwards=True.
    :param parallel_ray_normal_angle_tolerance: float, maximum angle between parallel ray normals and main hit normal.
        Used to ensure that the parallel rays hit a flat surface.
    :param refuse_downwards: bool, whether downward-facing hits (as defined by max_angle_with_z_axis) are allowed.
    :return: List of num_samples elements where each element is a tuple in the form of
        (cuboid_centroid, cuboid_up_vector, cuboid_rotation, {refusal_reason: [refusal_details...]}). Cuboid positions
        are set to None when no successful sampling happens within the max number of attempts. Refusal details are only
        filled if the debug_sampling flag is globally set to True.
    """
    aabb = obj.states[object_states.AABB].get_value()
    aabb_min = np.array(aabb[0])
    aabb_max = np.array(aabb[1])

    sampling_aabb_min = aabb_min - aabb_offset
    sampling_aabb_max = aabb_max + aabb_offset

    body_id = obj.get_body_id()

    results = [(None, None, None, defaultdict(list)) for _ in range(num_samples)]

    for i in range(num_samples):
        debug_markers = []

        # Sample the starting positions in advance.
        samples = sample_origin_positions(sampling_aabb_min, sampling_aabb_max, max_sampling_attempts,
                                          bimodal_mean_fraction, bimodal_stdev_fraction, axis_probabilities)

        refusal_reasons = results[i][3]

        # Try each sampled position in the AABB.
        for axis, is_top, start_pos in samples:
            # Compute the ray's destination using the sampling & AABB information.
            point_on_face = compute_ray_destination(axis, is_top, start_pos, aabb_min, aabb_max)

            # If we have a list of offset distances, pick the distance for this particular sample we're getting.
            this_cuboid_dimensions = cuboid_dimensions
            assert len(this_cuboid_dimensions), "Cuboid dimensions needs to be a sequence."
            if isinstance(this_cuboid_dimensions[0], list):
                assert len(this_cuboid_dimensions) == num_samples, "Need as many offsets as samples requested."
                this_cuboid_dimensions = this_cuboid_dimensions[i]
            assert len(this_cuboid_dimensions) == 3
            this_cuboid_dimensions = np.array(this_cuboid_dimensions)

            # Obtain the parallel rays using the direction sampling method.
            sources, destinations, grid = get_parallel_rays(
                start_pos, point_on_face, this_cuboid_dimensions[:2] / 2.)

            # Time to cast the rays.
            cast_results = p.rayTestBatch(rayFromPositions=sources, rayToPositions=destinations)

            # Check that all rays hit the object.
            if not check_rays_hit_object(cast_results, body_id, refusal_reasons):
                continue

            # Process the hit positions and normals.
            hit_positions = np.array([ray_res[3] for ray_res in cast_results])
            hit_normals = np.array([ray_res[4] for ray_res in cast_results])
            hit_normals /= np.linalg.norm(hit_normals, axis=1)[:, np.newaxis]

            center_idx = int(len(cast_results) / 2)
            center_hit_normal = hit_normals[center_idx]

            # Apply the padding to all the points.
            padding = bottom_padding * center_hit_normal
            hit_positions += padding

            center_hit_position = hit_positions[center_idx]

            # Reject anything facing more than 45deg downwards if requested.
            if refuse_downwards:
                if not check_hit_max_angle_from_z_axis(center_hit_normal, max_angle_with_z_axis, refusal_reasons):
                    continue

            # Check that none of the parallel rays' hit normal differs from center ray by more than threshold.
            if not check_hit_normal_similarity(center_hit_normal, hit_normals, parallel_ray_normal_angle_tolerance,
                                               refusal_reasons):
                continue

            # Now we use the cuboid's diagonals to check that the cuboid is actually empty.
            if not check_cuboid_empty(debug_markers, grid, center_hit_normal, hit_positions, refusal_reasons,
                                      this_cuboid_dimensions):
                continue

            # TODO: Check that the points are somewhat planar, e.g. fit a least-squares plane & check distances.

            # Compute the cuboid center
            cuboid_centroid = center_hit_position + center_hit_normal * this_cuboid_dimensions[2] / 2.

            # Compute a rotation from the default AABB to the sampled position.
            rotation = compute_rotation_from_grid_sample(grid, hit_positions, cuboid_centroid,
                                                         this_cuboid_dimensions)

            # We've found a nice attachment point. Continue onto next point to sample.
            results[i] = (cuboid_centroid, center_hit_normal, rotation.as_quat(), refusal_reasons)
            break

    if gibson2.debug_sampling:
        print("Sampling rejection reasons:")
        counter = Counter()

        for instance in results:
            for reason, refusals in instance[3].items():
                counter[reason] += len(refusals)

        print("\n".join("%s: %d" % pair for pair in counter.items()))

    return results


def compute_rotation_from_grid_sample(two_d_grid, hit_positions, cuboid_centroid, this_cuboid_dimensions):
    # TODO: Figure out if the normalization has any advantages.
    grid_in_planar_coordinates = two_d_grid.reshape(-1, 2)
    grid_in_object_coordinates = np.zeros((len(grid_in_planar_coordinates), 3))
    grid_in_object_coordinates[:, :2] = grid_in_planar_coordinates
    grid_in_object_coordinates[:, 2] = -this_cuboid_dimensions[2] / 2.
    grid_in_object_coordinates /= np.linalg.norm(grid_in_object_coordinates, axis=1)[:, None]

    sampled_grid_relative_vectors = hit_positions - cuboid_centroid
    sampled_grid_relative_vectors /= np.linalg.norm(sampled_grid_relative_vectors, axis=1)[:, None]

    # Grab the vectors that are nonzero in both
    nonzero_indices = np.logical_and(
        np.any(np.isfinite(grid_in_object_coordinates), axis=1),
        np.any(np.isfinite(sampled_grid_relative_vectors), axis=1))
    grid_in_object_coordinates = grid_in_object_coordinates[nonzero_indices]
    sampled_grid_relative_vectors = sampled_grid_relative_vectors[nonzero_indices]

    rotation, _ = Rotation.align_vectors(sampled_grid_relative_vectors, grid_in_object_coordinates)
    return rotation


def check_hit_normal_similarity(center_hit_normal, hit_normals, parallel_ray_normal_angle_tolerance, refusal_reasons):
    parallel_hit_main_hit_dot_products = np.clip(np.dot(hit_normals, center_hit_normal), -1.0, 1.0)
    parallel_hit_normal_angles_to_hit_normal = np.arccos(parallel_hit_main_hit_dot_products)
    all_rays_hit_with_similar_normal = np.all(
        parallel_hit_normal_angles_to_hit_normal < parallel_ray_normal_angle_tolerance)
    if not all_rays_hit_with_similar_normal:
        if gibson2.debug_sampling:
            refusal_reasons["parallel_hit_angle_off"].append(
                "normal %r, hit normals %r, hit angles %r" % (
                    center_hit_normal, hit_normals, parallel_hit_normal_angles_to_hit_normal))

        return False

    return True


def check_rays_hit_object(cast_results, body_id, refusal_reasons):
    hit_body_ids = [ray_res[0] for ray_res in cast_results]
    if not all(hit_body_id == body_id for hit_body_id in hit_body_ids):
        if gibson2.debug_sampling:
            refusal_reasons["missed_object"].append("hits %r" % hit_body_ids)

        return False

    return True


def check_hit_max_angle_from_z_axis(hit_normal, max_angle_with_z_axis, refusal_reasons):
    hit_angle_with_z = np.arccos(np.clip(np.dot(hit_normal, np.array([0, 0, 1])), -1.0, 1.0))
    if hit_angle_with_z > max_angle_with_z_axis:
        if gibson2.debug_sampling:
            refusal_reasons["downward_normal"].append("normal %r" % hit_normal)

        return False

    return True


def compute_ray_destination(axis, is_top, start_pos, aabb_min, aabb_max):
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

    return point_on_face


def check_cuboid_empty(debug_markers, grid, hit_normal, hit_positions, refusal_reasons,
                       this_cuboid_dimensions):
    # Get the indices of the corners
    y_dim = grid.shape[1]
    corner_indices = [0, y_dim - 1, -y_dim, -1]

    # Get sampled bottom corners & compute top corners.
    bottom_corner_positions = hit_positions[corner_indices]
    top_corner_positions = bottom_corner_positions + hit_normal * this_cuboid_dimensions[2]

    # Get all the top-to-bottom corner pairs. When we cast these rays, we check for two things: that the cuboid
    # height is actually available, and the faces & volume of the cuboid are unoccupied.
    top_to_bottom_pairs = list(itertools.product(top_corner_positions, bottom_corner_positions))

    # Get all the same-height pairs. These also check that the surfaces areas are empty.
    bottom_pairs = list(itertools.combinations(bottom_corner_positions, 2))
    top_pairs = list(itertools.combinations(top_corner_positions, 2))

    # Combine all these pairs, cast the rays, and make sure the rays don't hit anything.
    all_pairs = np.array(top_to_bottom_pairs + bottom_pairs + top_pairs)
    check_cast_results = p.rayTestBatch(rayFromPositions=all_pairs[:, 0, :], rayToPositions=all_pairs[:, 1, :])
    if not all(ray[0] == -1 for ray in check_cast_results):
        if gibson2.debug_sampling:
            refusal_reasons["cuboid_not_empty"].append("check ray info: %s" % check_cast_results)

        return False

    # Debug markers
    # TODO (Cem): Either make this possible to toggle on/off or delete it.
    # Kept for now for debugging cases that seem to spring up often.
    # from gibson2 import simulator
    # corner_color = np.concatenate([np.random.rand(3), [1]])
    # color = np.concatenate([np.random.rand(3), [1]])
    # colors = np.array([color for _ in range(len(hit_positions))])
    # colors[corner_indices] = corner_color
    # for idx, vec in enumerate(hit_positions):
    #     m = VisualMarker(
    #         rgba_color=colors[idx],
    #         radius=0.01,
    #         initial_offset=vec
    #     )
    #     simulator.SIM.import_object(m)
    #     debug_markers.append(m)

    return True
