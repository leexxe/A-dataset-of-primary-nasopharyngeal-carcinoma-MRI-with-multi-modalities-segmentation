import numpy as np
import vtkmodules.all as vtk
import nibabel as nib
import argparse

from pathlib import Path
from typing import Tuple


class MRIAnalyzer:
    def __init__(self):
        """
        Initializes the MRIAnalyzer class.
        """
        pass

    def load_nifti_image(self, patient_id: int, roi_idx: int) -> nib.Nifti1Image:
        """
        Loads a NIfTI image for a given patient ID and ROI index.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.
        roi_idx : int
            The index of the Region of Interest (ROI).

        Returns
        -------
        nib.Nifti1Image
            The loaded NIfTI image.
        """
        roi_names = ["ROI-T1.nii", "ROI-T2.nii", "ROI-CE-T1.nii"]
        path = Path(f"data/MRI-Segments/{patient_id:03d}/{roi_names[roi_idx]}")
        return nib.load(path)

    def create_binary_mask(self, image_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Creates a binary mask from the image data based on a computed threshold.

        Parameters
        ----------
        image_data : np.ndarray
            The image data from which to create the binary mask.

        Returns
        -------
        np.ndarray
            The binary mask.
        float
            The threshold value used to create the binary mask.
        """
        threshold_value = (np.min(image_data) + np.max(image_data)) / 2.0
        return (image_data > threshold_value).astype(np.uint8), threshold_value

    def vtk_image_from_mask(
        self, binary_mask: np.ndarray, nifti_image: nib.Nifti1Image
    ) -> vtk.vtkImageData:
        """
        Creates a VTK image from a binary mask and corresponding NIfTI image metadata.

        Parameters
        ----------
        binary_mask : np.ndarray
            The binary mask to convert.
        nifti_image : nib.Nifti1Image
            The NIfTI image for metadata extraction.

        Returns
        -------
        vtk.vtkImageData
            The resulting VTK image.
        """
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(binary_mask.shape)
        vtk_image.SetSpacing(
            nifti_image.header.get_zooms()[:3]
        )  # Set voxel spacing from the NIfTI header
        vtk_image.SetOrigin(
            nifti_image.affine[:3, 3]
        )  # Set origin from the NIfTI affine matrix
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Copy the binary mask data into the VTK image
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                for k in range(binary_mask.shape[2]):
                    vtk_image.SetScalarComponentFromFloat(
                        i, j, k, 0, binary_mask[i, j, k]
                    )
        return vtk_image

    def get_points(self, outer_surface: vtk.vtkPolyData) -> np.ndarray:
        """
        Extracts points from the outer surface of a VTK PolyData object.

        Parameters
        ----------
        outer_surface : vtk.vtkPolyData
            The VTK PolyData from which to extract points.

        Returns
        -------
        np.ndarray
            An array of points from the outer surface.
        """
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(outer_surface)  # Use the outer_surface as input
        surface_filter.Update()

        outer_boundary = surface_filter.GetOutput()
        points = outer_boundary.GetPoints()

        num_points = points.GetNumberOfPoints()

        points_np = np.array([points.GetPoint(i) for i in range(num_points)])
        return points_np

    def setup_rendering_pipeline(
        self, vtk_image: vtk.vtkImageData, threshold_value: float
    ) -> Tuple[
        vtk.vtkRenderWindowInteractor,
        vtk.vtkRenderer,
        vtk.vtkRenderWindow,
        vtk.vtkMarchingCubes,
    ]:
        """
        Sets up the VTK rendering pipeline for visualizing the segmented image.

        Parameters
        ----------
        vtk_image : vtk.vtkImageData
            The VTK image to be visualized.
        threshold_value : float
            The threshold value used for segmentation.

        Returns
        -------
        Tuple[vtk.vtkRenderWindowInteractor, vtk.vtkRenderer, vtk.vtkRenderWindow, vtk.vtkMarchingCubes]
            A tuple containing the render interactor, renderer, render window, and contouring filter.
        """
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_image)
        contour.SetValue(
            0, threshold_value
        )  # Adjust 'threshold_value' to match your specific segmentation

        # Create a renderer, render window, and interactor
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_interactor = vtk.vtkRenderWindowInteractor()
        render_interactor.SetRenderWindow(render_window)

        # Create the mapper and actor as before
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add the actor to the renderer
        renderer.AddActor(actor)

        # Set up the camera and render
        renderer.ResetCamera()
        render_window.Render()

        # Start the interaction
        # Note: Interaction is started but not shown in the script to prevent blocking
        def closeRenderWindow(caller, event):
            """
            Closes the render window on an event.

            Parameters
            ----------
            caller : vtk.vtkObject
                The caller of the event.
            event : str
                The event that triggered the call.
            """
            caller.TerminateApp()  # Stop the interactor loop
            render_window.Finalize()  # Clean up window resources
            render_interactor.SetRenderWindow(
                None
            )  # Disconnect the interactor from the window

        def setupCloseTimer():
            """
            Sets up a timer to close the render window automatically.

            Returns
            -------
            int
                The ID of the created timer.
            """
            timerId = render_interactor.CreateRepeatingTimer(1000)  # 1000 milliseconds
            render_interactor.AddObserver(
                "TimerEvent", lambda obj, event: closeRenderWindow(obj, event)
            )
            return timerId

        _ = setupCloseTimer()  # Setup the timer to close the window
        render_interactor.Start()

        return render_interactor, renderer, render_window, contour

    def compute_volume(self, contour: vtk.vtkMarchingCubes) -> float:
        """
        Computes the volume of the structure segmented by the contour.

        Parameters
        ----------
        contour : vtk.vtkMarchingCubes
            The contour filter used for segmentation.

        Returns
        -------
        float
            The computed volume.
        """
        mass_properties = vtk.vtkMassProperties()

        if (
            contour.GetOutput().GetNumberOfPoints() > 0
            and contour.GetOutput().GetNumberOfCells() > 0
        ):
            mass_properties.SetInputData(contour.GetOutput())
            mass_properties.Update()
            volume = mass_properties.GetVolume()
        else:
            print("No valid surface data to compute volume.")
            volume = 0

        return volume

    def compute_surface_area(
        self, contour: vtk.vtkMarchingCubes
    ) -> Tuple[float, vtk.vtkPolyData]:
        """
        Computes the surface area of the structure segmented by the contour and returns the outer surface.

        Parameters
        ----------
        contour : vtk.vtkMarchingCubes
            The contour filter used for segmentation.

        Returns
        -------
        float
            The computed surface area.
        vtk.vtkPolyData
            The outer surface of the segmented structure.
        """
        outer_surface = vtk.vtkPolyData()
        outer_surface.DeepCopy(contour.GetOutput())

        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputConnection(
            contour.GetOutputPort()
        )  # Use the contour as input

        clean_filter.Update()

        outer_surface = clean_filter.GetOutput()

        surface_properties = vtk.vtkMassProperties()
        if (
            outer_surface.GetNumberOfPoints() > 0
            and outer_surface.GetNumberOfCells() > 0
        ):
            surface_properties.SetInputData(outer_surface)
            surface_properties.Update()
            surface_area = surface_properties.GetSurfaceArea()
        else:
            print("No valid surface data to compute surface area.")
            surface_area = 0

        return surface_area, outer_surface

    def compute_max_diameter(
        self, points: np.ndarray
    ) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Computes the maximum diameter between any two points in the provided array.

        Parameters
        ----------
        points : np.ndarray
            An array of points.

        Returns
        -------
        float
            The maximum diameter.
        Tuple[np.ndarray, np.ndarray]
            The pair of points defining the maximum diameter.
        """
        num_points = len(points)
        max_diameter = 0
        max_diameter_points = (None, None)

        for i in range(num_points - 1):
            point1 = points[i]
            distances = np.linalg.norm(points[i + 1 :] - point1, axis=1)
            max_distance_index = np.argmax(distances)
            max_distance = distances[max_distance_index]

            if max_distance > max_diameter:
                max_diameter = max_distance
                max_diameter_points = (point1, points[i + 1 :][max_distance_index])

        return max_diameter, max_diameter_points

    def compute_surface_regularity(self, volume: float, surface_area: float) -> float:
        """
        Computes the surface regularity index of a volume, given its volume and surface area.

        Parameters
        ----------
        volume : float
            The volume of the structure.
        surface_area : float
            The surface area of the structure.

        Returns
        -------
        float
            The surface regularity index.
        """
        # Compute the equivalent radius of a sphere with the same surface area
        equivalent_radius = np.sqrt(surface_area / (4 * np.pi))

        # Calculate the volume of the sphere with the equivalent radius
        spherical_tumor_volume = (4 / 3) * np.pi * (equivalent_radius**3)

        surface_regularity = volume / spherical_tumor_volume

        return surface_regularity

    def compute_3d_values(self, patient_id: int, roi_idx: int) -> None:
        """
        Computes and prints 3D values (volume, surface area, maximum diameter, surface regularity)
        for a given patient's ROI.

        Parameters
        ----------
        patient_id : int
            The ID of the patient.
        roi_idx : int
            The index of the Region of Interest (ROI).
        """
        nifti_image = self.load_nifti_image(patient_id, roi_idx)
        image_data = nifti_image.get_fdata()
        binary_mask, threshold_value = self.create_binary_mask(image_data)
        vtk_image = self.vtk_image_from_mask(binary_mask, nifti_image)

        render_interactor, renderer, render_window, contour = (
            self.setup_rendering_pipeline(vtk_image, threshold_value)
        )

        volume = self.compute_volume(contour)
        surface_area, outer_surface = self.compute_surface_area(contour)

        surface_regularity = self.compute_surface_regularity(volume, surface_area)

        points_np = self.get_points(outer_surface)

        max_diameter, _ = self.compute_max_diameter(points_np)

        print(
            f"Volume: {volume} cubic units \n Surface Area: {surface_area} square units \n Max Diameter: {max_diameter} \n Surface Regularity: {surface_regularity}"
        )

        # Cleanup to prevent memory leaks
        renderer.RemoveAllViewProps()
        render_interactor.RemoveObservers("SomeVTKEvent")
        render_interactor.TerminateApp()
        render_window.Finalize()


def main():
    parser = argparse.ArgumentParser(description="Compute 3D values from MRI segments.")
    parser.add_argument("patient_id", type=int)
    parser.add_argument("roi_idx", type=int)

    args = parser.parse_args()

    patient_id = args.patient_id
    roi_idx = args.roi_idx

    analyzer = MRIAnalyzer()
    analyzer.compute_3d_values(patient_id, roi_idx)


if __name__ == "__main__":
    main()
