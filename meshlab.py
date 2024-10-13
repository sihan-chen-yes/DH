import pymeshlab
import os
import argparse

def poisson_reconstruction(path):
    '''
    use meshlab default parameters to do Poisson Reconstruction given point clouds
    Parameters
    ----------
    path

    Returns
    -------

    '''
    # MeshLab
    ms = pymeshlab.MeshSet()
    print("loading mesh...")
    ms.load_new_mesh(os.path.join(path, "point_cloud.ply"))

    print("estimating normal...")
    ms.compute_normal_for_point_clouds()

    print("Poisson reconstructing...")
    ms.generate_surface_reconstruction_screened_poisson(
        threads=32
    )

    ms.meshing_invert_face_orientation(forceflip=True)
    print("Saving mesh...")
    file_name = os.path.join(path, "output_mesh.ply")
    ms.save_current_mesh(
        file_name=file_name,
        binary=False
    )
    print("Poisson mesh saved at {}".format(file_name))

    # pcd = o3d.io.read_point_cloud(os.path.join(path, "point_cloud.ply"))
    # self.gaussians.load_ply(os.path.join(path, "point_cloud.ply"))
    # # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.k))
    #
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.depth)
    # # densities = np.asarray(densities)
    # # density_threshold = np.percentile(densities, 5)
    # # vertices_to_remove = densities < density_threshold
    # # mesh.remove_vertices_by_mask(vertices_to_remove)
    # o3d.io.write_triangle_mesh(os.path.join(path, "output_mesh.ply"), mesh)

    # print("Poisson mesh saved at {}".format(os.path.join(path, "output_mesh.ply")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Poisson Reconstruction Script")

    parser.add_argument('--name', type=str, required=True, help="Name of exp")
    parser.add_argument('--frame', type=int, required=True, help="frame to reconstruct")


    args = parser.parse_args()

    path = os.path.join("exp", args.name, "point_cloud/frame_{}".format(args.frame))

    poisson_reconstruction(path)