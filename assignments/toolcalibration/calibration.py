import numpy as np


def pivot_calibration(transforms):
    """ Pivot calibration
    Keyword arguments:
    transforms -- A list of 4x4 transformation matrices from the tracking system (Fi)
    returns    -- The calibration matrix T (p_t in homogeneous coordinate form),
                  where the vector p_t, is the offset from any transform, Fi, to the pivot point
    """

    ## TODO: Implement pivot calibration as discussed in the lecture
    
    T = np.eye(4)

    try:

        # 1. [TRANSFORM MATRIX DECOMPOSITION] F_I = (R_i, p_i)
        F_i = np.stack(transforms)

        # 1.2 [TRANSLATION]: p' = M * p --> [d_x d_y d_z]
        p_i = F_i[:,0:3,3]
    
        # 1.2 [ROTATION & SCALE]: p' = M * p --> [[m_xx m_xy m_xz] ...[m_zy .. m_zz]]
        R_i = F_i[:, 0:3, 0:3]
        
        # 2.1 [Ax = b]: b = p_i 
        b =  (-1) * p_i

        # we simplify b matrix dimension by reducing from 2D --> 1D
        b = b.reshape((b.shape[0]*b.shape[1],1))

        # 2.2 [Ax = b]: A = [[R_i  -I] ... [R_n  -I]]
        I =  np.eye(3) * (-1)

        # we stack 259 or index i len of identity matrix using numpy.tile
        I = I.reshape((1,3,3)).repeat(R_i.shape[0],axis=0)

        # we simplify I matrix dimension by reducing from 3D --> 2D
        I = I.reshape(I.shape[0]*I.shape[1],I.shape[2])

        # we simplify R_i matrix dimension by reducing from 3D --> 2D
        R_i = R_i.reshape((R_i.shape[0]*R_i.shape[1], R_i.shape[2]))

        A = np.hstack((R_i,I))

        # 3.  [SVD]: A = UΣV^T, we put full_matrices as False, we want to keep the no.of matrices (last dimension)
        U,S,V = np.linalg.svd(A,full_matrices=False)

        # 3.1 [SVD]: Σ^−1
        S_inv = np.diag(1 / S)
        
        # 3.2 [SVD]: x = V * Σ^−1 * U^T * b
        x = V.T @ S_inv @ U.T @ b

        # 4. x = [p_t p_p]
        p_t = x[0:3].flatten()
        
        T[0:3,3] = p_t

    except Exception as e:
        print("[ERROR.pivot_calibration]: ",e)

    return T


def calibration_device_calibration(camera_T_reference, camera_T_tool, reference_P_pivot):
    """ Tool calibration using calibration device
    Keyword arguments:
    camera_T_reference -- Transformation from the reference (calibration device) to the camera
    camera_T_tool      -- Transformation from the tool to the camera
    reference_P_pivot  -- A pivot point on the calibration device reference (rigid body),
                          where the tip of the instrument is located for calibration
    returns            -- The tool tip location (p_t or reference_P_pivot) and the
                          calibration matrix (T), i.e. the tool tip location
                          (reference_P_pivot) relative to the tracked tool (camera_T_tool)
    """
    
    ## TODO: Implement a calibration method which uses a calibration device

    # MATH: I = ABCD => B = (A^-1 D^-1) C^-1
    # A = camera_T_tool
    # D = reference_T_camera
    # C = tip_T_reference
    # GOAL: find calibration matrix tool_T_tip = (reference_t_camera) 
    
    # 1. tool_t_tip = (camera_T_tool^-1 * tip_T_reference^-1) * reference_T_camera^-1
    T = np.linalg.inv(camera_T_tool) @ camera_T_reference @ reference_P_pivot 
    
    return T
