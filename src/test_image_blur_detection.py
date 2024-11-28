import pytest
import os
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from image_blur_detection import detect_image_blur#, detect_image_blur_helper #old way of doing it

@pytest.fixture
def mock_image():
    """Simulate a grayscale image with a specific variance."""
    mock_image = MagicMock()
    mock_image.var.return_value = 60  # Variance above threshold
    return mock_image

@pytest.fixture
def temp_folder(tmp_path):
    """Create a temporary directory with some test image files."""
    test_folder = tmp_path / "test_images"
    test_folder.mkdir()
    (test_folder / "clear_image.png").touch()  # Create an empty file
    (test_folder / "blurry_image.jpeg").touch()
    return test_folder

# @patch("cv2.imread", return_value=MagicMock())
# @patch("cv2.cvtColor", return_value=MagicMock())
# @patch("cv2.Laplacian")
# @patch("builtins.open", new_callable=mock_open)
# def test_detect_image_blur(mock_open_file, mock_laplacian, mock_cvtColor, mock_imread, mock_image):
#     # Mock the Laplacian filter and its variance
#     mock_laplacian.return_value = mock_image
#     mock_image.var.return_value = 40  # Variance below threshold

#     # Call the function
#     detect_image_blur("path/to/image.png", "path/to/folder")
    
#     # Check if the output file is written correctly
#     expected_path = os.path.join(os.getcwd(), "image_blur_results.txt")
#     mock_open_file.assert_called_once_with(expected_path, "w")


# # Test for detect_image_blur_helper
# @patch("os.walk")
# @patch("image_blur_detection.detect_image_blur")
# """
# def test_detect_image_blur_helper(mock_detect_blur, mock_os_walk, temp_folder):
#     # Mock os.walk to return test folder structure
#     mock_os_walk.return_value = [
#         (str(temp_folder), [], ["clear_image.png", "blurry_image.jpeg"]),
#     ]
    
#     # Call the helper function
#     detect_image_blur_helper(str(temp_folder))
    
#     # Check that detect_image_blur was called for each valid image
#     mock_detect_blur.assert_any_call(
#         os.path.join(str(temp_folder), "clear_image.png"), str(temp_folder)
#     )
#     mock_detect_blur.assert_any_call(
#         os.path.join(str(temp_folder), "blurry_image.jpeg"), str(temp_folder)
#     )

# # Additional edge case: Folder does not exist
# def test_detect_image_blur_helper_invalid_folder():
#     with pytest.raises(SystemExit):  # Expect the function to quit
#         detect_image_blur_helper("invalid/folder/path")
# """

@patch("cv2.imread", return_value=MagicMock())
@patch("cv2.cvtColor", return_value=MagicMock())
@patch("cv2.Laplacian")
@patch("builtins.open", new_callable=mock_open)
def test_detect_image_blur(mock_open_file, mock_laplacian, mock_cvtColor, mock_imread, mock_image):
    
    # Test Case 1: image below threshold must return true
    
    # Mock the Laplacian filter and its variance
    mock_laplacian.return_value = mock_image
    mock_image.var.return_value = 40  # Variance below threshold
    
    blur_check = detect_image_blur(mock_image)
    
    assert(blur_check is True)
    
    # Test case 2: image above threshold must return to be false
    mock_laplacian.return_value = mock_image
    mock_image.var.return_value = 80  # Variance above threshold
    
    blur_check = detect_image_blur(mock_image)
    
    assert(blur_check is False)
    
    
    # Test case 3: wrong type for image_path
    with pytest.raises(TypeError, match= "image path must be type str or pathlib.Path"):
        detect_image_blur(image_path=4)
        
        # Test case 4: image_path is None
    with pytest.raises(TypeError, match= "image path must be type str or pathlib.Path"):
        detect_image_blur(image_path=None)    