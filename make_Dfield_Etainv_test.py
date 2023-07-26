def make_Dfield_Etainv(Etainv, coord, cell):
    """ Compute the softness kernel at finite D from the one at finite E

    Args:
        eta_inv (tensor): a [B, N, N] tensor containing the softness kernel
        coord (tensor) : a [B, N, N] tensor containing the atomic positions
        cell (tensor) : a [B, 3, 3] tensor containing the cell parameters

    Returns:
        chi (tensor): a [B, N, N] tensor containing the linear response kernel
    """
    from numpy import pi
    Omega = cell[0][0] * cell[1][1] * cell[2][2]
    fourpi = 4 * pi
#    for i in range(3):
#        if D_applied(i):
#            R = coord[:,i]
#            AR   = tf.einsum('bi,bij->bi', R, eta_inv)
#            ARRA = tf.einsum('bi,bj->bij', AR, AR) * fourpi / Omega
#            RAR  = tf.einsum('bi,bi->b', R, AR) * fourpi / Omega
#            eta_inv  =  Ainv - ARRA/(1.+RAR[:,None,None])
            
    R = coord[2,:]
    RR = tf.einsum('i,j->ij', R, R)
    RRA = tf.einsum('bi,bij->bi', RR, Etainv)
    AR   = tf.einsum('bi,bij->bi', R, Etainv)
    ARRA = tf.einsum('bi,bj->bij', AR, AR) * fourpi / Omega
    RAR  = tf.einsum('bi,bi->b', R, AR) * fourpi / Omega
    eta_inv  =  Etainv + (1- RRA/(1.+RAR[:,None,None]))
    return eta_inv