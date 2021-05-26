
"""
Taken and Customized from:
            https://github.com/KushalBKusram/WaymoDataToolkit
"""
"""
As step 1
Extract images and labels from segment
"""
import time
import threading
from datetime import timedelta
import util_image_label as WODKit

if __name__=="__main__":

    #Path where the input and output folders are available
    main_path = "/opt/pie/PIEPredict/WAYMO"
    #main_path = "/home/eyuell/Desktop/ForD/WAYMO"

    segments_dir = main_path + "/input/segment"
    output_dir = main_path + "/output"

    start = time.time()

    print('Processing . . .  \r')

    toolkit = WODKit.ToolKit(training_dir=segments_dir, save_dir=output_dir)

    # clear images, labels, consolidation and ped lists from previous execution
    toolkit.refresh_dir(dir="{}/camera/images".format(output_dir))
    toolkit.refresh_dir(dir="{}/camera/labels".format(output_dir))
    toolkit.refresh_dir(dir="{}/consolidation".format(output_dir))
    open("{}/camera/ped_frames_file.txt".format(output_dir), 'w').write("{}".format(""))

    # prepare for progress monitoring
    i = 0
    segments = toolkit.list_training_segments()
    size = len(segments)

    # Process through all segments
    for segment in segments:
        i += 1
        #print('Processing . . .         {}/{} ({:.2f}%) \r'.format(i, size, (i/size) * 100))
        print('Processing . . . segment {} of {}\r'.format(i, size))

        threads = []

        toolkit.assign_segment(segment, i)

        t1 = threading.Thread(target=toolkit.extract_camera_images)
        t1.start()
        threads.append(t1)

        for thread in threads:
            thread.join()

        toolkit.consolidate()

        #break # if only for one segment

    # Concluding
    end = time.time()
    elapsed = end - start
    t_del = timedelta(seconds=elapsed)

    if i == size:
        print('\nDone! Duration= {}\n'.format(t_del))
    else:
        print('\nExited! Duration= {}\n'.format(t_del))
