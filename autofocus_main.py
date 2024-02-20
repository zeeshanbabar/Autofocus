from collections.abc import Callable
from enum import Enum, auto
from time import sleep
from threading import Thread
from queue import Empty, Full, Queue

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer


from standa import StandaController

sc = StandaController()

# -----------------------------------------------------------------------------------------------------------------


class FigureOfMerit:
    def __init__(self) -> None:
        pass

    def calculate_figure_of_merit(self, img: np.ndarray) -> float:
        image = cv.medianBlur(img, 3)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        laplacian = cv.Laplacian(gray, cv.CV_64F)
        return float(laplacian.var())

    def fom_worker(self, in_queue: Queue, out_queue: Queue):
        while True:
            # Block until we get an image
            img, pos = in_queue.get(block=True)

            fom = self.calculate_figure_of_merit(img)
            # print(f"{fom=}")

            # Output result
            out_queue.put((fom, pos), block=False)


class Commands(Enum):
    """Commands to be put in the cmd_queue for CamProcess"""
    START = auto()
    STOP = auto()
    IDLE = auto()
    END = auto()


class InitialPosition:
    def __init__(self) -> None:  # starting point
        self.standa_x_init = 1192036
        self.standa_y_init = 2585917
        self.standa_z_init = 859728


class CamProcess(Thread):
    def __init__(self, img_queue: Queue, cmd_queue: Queue, get_z: Callable[[], float], **kwargs):

        super().__init__(kwargs=kwargs)

        self.img_queue = img_queue
        self.cmd_queue = cmd_queue
        self.get_z = get_z

        self.init: bool = False
        self.running: bool = True
        self.output: bool = False

        self.frame_count: int = 0

    def run(self):
        from pythorcam.pythorcam import ThorCam
        # Connect to camera
        with ThorCam(output_bitdepth=ThorCam.BitDepth.RGB_8) as camera:
            camera.exposure_time_ms = 1.5
            window_name: str = f"Thorlabs {camera.camera.model}, serial #: {camera.camera.serial_number}"
            cv.namedWindow(window_name, cv.WINDOW_FREERATIO)
            with camera.get_stream() as stream:
                self.init = True
                while True:
                    try:
                        cmd: Commands = self.cmd_queue.get(block=False)
                        match cmd:
                            case Commands.START:
                                self.frame_count = 0
                                self.running = True
                                self.output = True
                            case Commands.STOP:
                                self.running = False
                                self.output = False
                            case Commands.IDLE:
                                self.running = True
                                self.output = False
                            case Commands.END:
                                break

                    except Empty:
                        pass

                    # Do nothing if not running
                    if not self.running:
                        sleep(0.001)
                        continue

                    img = stream.get_frame()
                    frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    pos = self.get_z()

                    self.frame_count += 1

                    # Update GUI
                    cv.imshow(window_name, frame)

                    frame_delay_ms = 1

                    # Check to close
                    if (cv.waitKey(frame_delay_ms) == ord("q")):
                        break

                    # Send away frame
                    if self.output:
                        try:
                            self.img_queue.put((img, pos), block=False)
                        except Full:
                            print("Queue full, ignoring")


def main():
    start_time = timer()
    fom_c = FigureOfMerit()

    z_positions = []

    cmd_queue = Queue()
    img_queue = Queue()
    fom_queue = Queue()

    # Create processes
    camera_proc = CamProcess(img_queue=img_queue, cmd_queue=cmd_queue, get_z=sc.get_z, daemon=True)
    fom_thread = Thread(target=fom_c.fom_worker, args=(img_queue, fom_queue), daemon=True)

    # Start Process
    camera_proc.start()
    fom_thread.start()
    while not camera_proc.init:
        sleep(0.01)
    print("camera stream has started")

    corners = [
        (1, 1032569, 2040889),
        (2, 1010377, 2922801),
        (3, 1809481, 2915910),
        (4, 1840658, 2053684),
    ]
    (min_z, max_z) = (871658, 844836)

    # Visit corners
    for i, x_pos, y_pos in corners:
        # Mov to start position
        print(f"Going to Corner {i}")
        sc.x.axis.move(x_pos)
        sc.y.axis.move(y_pos)
        print(f"Reached Corner {i}. Moving to max z position")
        sc.z.axis.set_speed(50)
        sc.z.axis.move(max_z)
        print("Max z position reached")

        # Start acquiring imgs
        cmd_queue.put(Commands.START)
        while not camera_proc.output:
            sleep(0.01)  # Wait until camera is streaming

        # Perform movement sweep
        sc.z.axis.set_speed(10)
        print("Moving to bottom z position for sweap")
        sc.z.axis.move(min_z)  # blocking
        print("Sweap Completed")

        # Stop image acquisition
        cmd_queue.put(Commands.IDLE)
        while camera_proc.output:
            sleep(0.01)  # Wait until sweep finishes

        sweep: list[tuple[float, float]] = [fom_queue.get() for _ in range(camera_proc.frame_count)]

        sweep_fom, sweep_pos = zip(*sweep)

        if len(sweep_fom) == 0:
            print("No fom...")
            best_pos = min_z
        else:
            best_pos = sweep_pos[np.argmax(sweep_fom)]
            sc.z.axis.set_speed(50)
            print("Moving to best focussed position")
            sc.z.axis.move(best_pos)
            z_positions.append(best_pos)

            plt.plot(sweep_pos, sweep_fom)
            plt.xlabel("Pos")
            plt.ylabel("Variance")
            plt.grid(True, which="both")
            plt.vlines(best_pos, *plt.ylim(), colors=["red"], linestyles="dashed")
            plt.show()  # Block until closed

    print(f"All the four corner positions are {z_positions}")
    end_time = timer()
    print(f"The totaltime to complete the process is {end_time - start_time}")

    cmd_queue.put(Commands.END)
    camera_proc.join()


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        sc.x.stop()
        sc.y.stop()
        sc.z.stop()
