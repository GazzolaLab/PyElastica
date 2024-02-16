// POV-Ray 3.6 / 3.7 Scene File "Ribbon_Cable_1.pov"
// author: Friedrich A. Lohmueller, Sept-2009/Jan-2011
// email: Friedrich.Lohmueller_at_t-online.de
// homepage: http://www.f-lohmueller.de
//--------------------------------------------------------------------------
#version 3.6; // 3.7;
global_settings{ assumed_gamma 1.5 }
#default{ finish{ ambient 0.1 diffuse 0.9 }}
#include "colors.inc"
#include "textures.inc"
#include "glass.inc"
#include "metals.inc"
#include "golds.inc"
#include "stones.inc"
#include "woods.inc"
#include "shapes.inc"
#include "shapes2.inc"
#include "functions.inc"
#include "math.inc"
#include "transforms.inc"

background{ color rgb<1,1,1>}

#declare Texture1  = texture {
                               pigment{ color rgb<1,0,0>}
                               finish { phong 1}
                             }
