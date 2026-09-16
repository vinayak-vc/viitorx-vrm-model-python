#!/usr/bin/env python3
"""F-19 section 7 - read the OAK-D BNO086 accelerometer to establish the sign convention ONCE,
so f19_level.py can report a signed pitch/roll rather than a magnitude."""
import numpy as np
import depthai as dai

p = dai.Pipeline()
imu = p.create(dai.node.IMU)
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
x = p.create(dai.node.XLinkOut)
x.setStreamName("imu")
imu.out.link(x.input)

with dai.Device(p) as dev:
    q = dev.getOutputQueue("imu", maxSize=50, blocking=False)
    acc = []
    while len(acc) < 200:
        for pk in q.get().packets:
            a = pk.acceleroMeter
            acc.append((a.x, a.y, a.z))
    A = np.array(acc)
    m = A.mean(axis=0)
    s = A.std(axis=0)
    print("samples            : %d" % len(A))
    print("accel mean (m/s^2) : x %+7.3f  y %+7.3f  z %+7.3f" % tuple(m))
    print("accel std          : x %7.3f  y %7.3f  z %7.3f" % tuple(s))
    print("magnitude          : %.3f  (9.807 = at rest, so the reading is pure gravity)"
          % np.linalg.norm(m))
    print()
    print("DepthAI camera frame: X right, Y down, Z forward (optical axis).")
    print("dominant axis      : %s" % "XYZ"[int(np.argmax(np.abs(m)))])
