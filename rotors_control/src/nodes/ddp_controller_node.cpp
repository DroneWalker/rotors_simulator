/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ros/ros.h>
#include <mav_msgs/default_topics.h>

#include "lee_position_controller_node.h"

#include "rotors_control/parameters_ros.h"

#include <iomanip>
#include <iostream>
#include "viewer.h"
#include "lib/views/view.h"
#include "qrotorview.h"
#include "lib/utils/so3.h"
#include "lib/utils/se3.h"
#include "lib/utils/utils.h"
#include "lib/utils/function.h"
#include "lib/systems/manifolds/rn.h"
#include "lib/systems/manifolds/manifold.h"
#include "lib/systems/manifolds/body3dmanifold.h"
#include "system.h"
#include "lib/systems/qrotor.h"
#include "lib/systems/body3d.h"
#include "lib/utils/params.h"
#include "lib/systems/costs/body3dcost.h"
#include "lib/systems/costs/cost.h"
#include "unistd.h"


//#define USE_SDDP

#ifdef USE_SDDP
#include "lib/algos/sddp.h"
#else
#include "lib/algos/ddp.h"
#endif

/////////////////////// GCOP STUFF //////////////////
using namespace gcop;
using namespace Eigen;
using namespace std;

typedef Ddp<Body3dState, 12, 4> QrotorDdp;

Params params;

void solver_process(Viewer* viewer)
{
//    if (viewer)
//        viewer->SetCamera(43.25, 94, -1.15, -0.9, -3.75);


    int N = 32;      // discrete trajectory segments
    double tf = 2;   // time-horizon

    params.GetInt("N", N);
    params.GetDouble("tf", tf);

    double h = tf/N; // time-step

    // system
    Qrotor sys;


    Body3dState x0;
    VectorXd qv0(12);
    params.GetVectorXd("x0", qv0);
    SO3::Instance().q2g(x0.R, qv0.head(3));
    x0.p = qv0.segment<3>(3); x0.w = qv0.segment<3>(6); x0.v = qv0.tail<3>();

    Body3dState xf;
    VectorXd qvf(12);
    params.GetVectorXd("xf", qvf);
    SO3::Instance().q2g(xf.R, qvf.head(3));
    xf.p = qvf.segment<3>(3); xf.w = qvf.segment<3>(6); xf.v = qvf.tail<3>();

    Body3dCost<4> cost(sys, tf, xf);

    VectorXd Q(12);
    VectorXd R(4);
    VectorXd Qf(12);

    params.GetVectorXd("Q", Q);
    params.GetVectorXd("R", R);
    params.GetVectorXd("Qf", Qf);

    cost.Q = Q.asDiagonal();
    cost.R = R.asDiagonal();
    cost.Qf = Qf.asDiagonal();

    // times
    vector<double> ts(N+1);
    for (int k = 0; k <= N; ++k)
        ts[k] = k*h;

    // states
    vector<Body3dState> xs(N+1, x0);

    // initial controls (e.g. hover at one place)
    vector<Vector4d> us(N);
    for (int i = 0; i < N; ++i) {
        us[i].head(3).setZero();
        us[i][3] = 9.81*sys.m;
    }

    QrotorDdp ddp(sys, cost, ts, xs, us);
    ddp.mu = .01;

//    QrotorView view(sys, &ddp.xs);
//    if (viewer)
//        viewer->Add(view);
//        cout << "Update View" << endl;


    struct timeval timer;
    //  ddp.debug = false; // turn off debug for speed

    for (int i = 0; i < 100; ++i) {
        timer_start(timer);
        ddp.Iterate();
        long te = timer_us(timer);
        cout << "Iteration #" << i << ": took " << te << " us." << endl;
        //    getchar();
    }

    cout << "done!" << endl;
    cout << "Click on window and press 'a' to animate." << endl;
    while(1)
        usleep(10);
}


#define DISP

int main(int argc, char** argv)
{
    if (argc > 1)
        params.Load(argv[1]);
    else
        params.Load("/home/charris/rotors_ws/rotors_simulator/rotors_control/src/library/qrotor.cfg");

    ros::init(argc, argv, "ddp_position_controller_node");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
//    rotors_control::LeePositionControllerNode lee_position_controller_node(nh, private_nh);
    cout << "Running Solver!" << endl;
    solver_process(0);
    ros::spin();



    return 0;
}
