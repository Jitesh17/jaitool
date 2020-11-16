# from jaitool.annotation.NDDS import NDDS_Dataset
from annotation_utils.ndds.structs import NDDS_Dataset

from pyjeasy.image_utils.edit import merge_colors
from typing import List, Tuple, Union

def merge_objects_by_iscolor(
    ndds_path: str=None,
    color_to: Union[int, list, tuple]=None, 
    color_from: list=None, 
    except_colors: List[list]=None, 
    except_bg_colors: bool=False, 
    show_preview: bool=False, 
    write_image_path: str=None, 
    write_image: bool=False, 
    verbose: bool=False
    ): 
    
    """
    Change specific colors to a certain color in the given image.
    
    Input:
    image_path, color_to, 
    color_from: list=None, 
    except_colors: list=None,
    except_bg_colors: bool=False, # Backgroung is decided by the color region with largest area
    show_image: bool=False, 
    write_image_path: src=None
    """
    # make_dir_if_not_exists(os.path.abspath(os.path.join(coco_data_dir, '../..')))
    # make_dir_if_not_exists(os.path.abspath(os.path.join(coco_data_dir, '..')))
    # make_dir_if_not_exists(coco_data_dir)
    # Load NDDS Dataset
    ndds_dataset = NDDS_Dataset.load_from_dir(
        json_dir=ndds_path,
        show_pbar=True
    )

    # Fix NDDS Dataset naming so that it follows convention. (This is not necessary if the NDDS dataset already follows the naming convention.)
    for frame in ndds_dataset.frames:
        is_path = frame.img_path.split('.')[0]+'.is.'+frame.img_path.split('.')[-1]
        if write_image:
            if write_image_path == None:
                output_image_path = is_path
            else:
                output_image_path = write_image_path + '/' + frame.img_path.split('/')[-1]
        else:
            output_image_path = None
        quit = merge_colors(
            image_path=is_path,         
            color_to=color_to, 
            color_from=color_from, 
            except_colors=except_colors,
            except_bg_colors=except_bg_colors, 
            show_preview=show_preview, 
            write_image_path=output_image_path, 
            verbose=verbose)
        if quit:
            break

if __name__ == "__main__":

    folder_name = f'b8'
    ndds_path = f'/home/jitesh/3d/data/UE_training_results/{folder_name}'
    # main(ndds_path=ndds_path, except_colors=[[255, 255, 252]], show_preview=True)