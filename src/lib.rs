#[macro_use] extern crate error_chain;
extern crate rand;
extern crate num;
use errors::*;
use rand::Rng;
use num::Complex;
use std::cmp::Ordering;
use rand::distributions::normal::StandardNormal;
use num::{Float, NumCast};

mod errors {
    // Create the Error, ErrorKind, ResultExt, and Result types
    error_chain!{}
}

/* TODO
    - use faster random number generator
    - better fitness and sampling functions for metropolis
*/

pub fn fit_to_ratio<F: Float>(ratio: F, x_min: &mut F, x_max: &mut F, y_min: &mut F, y_max: &mut F) -> Result<()> {
    let view_ratio = (*x_max - *x_min) / (*y_max - *y_min);
    let two = NumCast::from(2).unwrap();

    match ratio.partial_cmp(&view_ratio).chain_err(|| "problem with view and/or screen coordinates")? {
        Ordering::Less => {
            let diff_y = (*y_max - *y_min) / ratio;
            let center_y = (*y_max - *y_min) / two + *y_min;
            *y_min = center_y - diff_y / two;
            *y_max = center_y + diff_y / two;
        },
        Ordering::Greater => {
            let diff_x = (*x_max - *x_min) * ratio;
            let center_x = (*x_max - *x_min) / two + *x_min;
            *x_min = center_x - diff_x / two;
            *x_max = center_x + diff_x / two;
        },
        Ordering::Equal => {}
    };

    Ok(())
}


pub fn gen_buddhabrot(size_x: usize, size_y:usize, x_min: f32, x_max:f32, y_min: f32, y_max: f32, min_iteration: u32, max_iteration: u32, num_sample: u32, image: &mut [u16]) -> Result<(u16)> {
    let mut rng = rand::thread_rng();
    //let seed: &[_] = &[1, 2, 3, 4];
    //let mut rng: StdRng = SeedableRng::from_seed(seed);


    let mut max = 0u16;

    'main_loop: for _ in 0..num_sample {
        // generate a complex from a uniform rectangular distribution covering the mandelbrot set
        let c = Complex::new((rng.gen::<f32>()-0.5)*4f32, (rng.gen::<f32>()-0.5)*4f32);

        // check if c is in the cardioid || in the period-2 bulb
        let c_im_powi_2 = c.im.powi(2);
        let q = (c.re-0.25).powi(2)+ c_im_powi_2;
        if q*(q+(c.re-0.25)) < 0.25 * c_im_powi_2 || (c.re + 1.0).powi(2) + c_im_powi_2 < 0.0625{
            continue;
        }

        // find out if c is in the mandelbrot set
        let mut z = Complex::new(0f32,0f32);
        let mut i = 0u32;
        let mut escaped: bool = false;
        
        while i < std::cmp::min(max_iteration, 2000) {
            z = z*z + c;
            i += 1;
            if (z.re*z.re+z.im*z.im) >= 4f32 {escaped = true; break}
        }

        if i == max_iteration {continue}

        if !escaped {
            'fix_sized_outer_loop: while i <= max_iteration - 4000 {
                let old_z = z;
                for j in 0..4000 {
                    z = z*z + c;
                    if z == old_z {continue 'main_loop}
                    if (z.re*z.re+z.im*z.im) >= 4f32 {
                        escaped = true;
                        i += j;
                        break 'fix_sized_outer_loop
                    }
                }
                i += 4000;
            }
        }

        if i == max_iteration {continue}

        if !escaped {
            while i < max_iteration {
                z = z*z + c;
                i += 1;
            }
            if (z.re*z.re+z.im*z.im) >= 4f32 {escaped = true}
        }
        

        // if c escaped before max_iteration (therefore is NOT in the mandelbrot set)
        if i > min_iteration && escaped {
            // retrace the path of z
            z = c;
            while (z.re*z.re+z.im*z.im) < 4f32 {
                let coord_x = ((z.re-x_min)/(x_max-x_min)*size_x as f32) as i32;
                let coord_y = ((z.im-y_min)/(y_max-y_min)*size_y as f32) as i32;
                
                if coord_x >= 0 && coord_x < size_x as i32 {
                    if coord_y >= 0 && coord_y < size_y as i32 {
                        let new = image[coord_y as usize *size_x+coord_x as usize].saturating_add(1);
                        if new > max {
                            max = new;
                        }
                        image[coord_y as usize * size_x + coord_x as usize] = new;
                    }

                    // trace the symetric on axe x
                    let coord_y = ((-z.im-y_min)/(y_max-y_min)*size_y as f32) as i32;
                    if coord_y >= 0 && coord_y < size_y as i32 {
                        let new = image[coord_y as usize *size_x+coord_x as usize].saturating_add(1);
                        if new > max {
                            max = new;
                        }
                        image[coord_y as usize * size_x + coord_x as usize] = new;
                    }
                }
                
                z = z*z + c;
            }
        }
    }

    return Ok(max);
}

pub fn gen_buddhabrot_metropolis(size_x: usize, size_y:usize, x_min: f64, x_max:f64, y_min: f64, y_max: f64, min_iteration: u32, max_iteration: u32, num_sample: u32, image: &mut [u16]) -> Result<(u16)> {
    let mut rng = rand::thread_rng();
    //let seed: &[_] = &[1, 2, 3, 4];
    //let mut rng: StdRng = SeedableRng::from_seed(seed);

    let std_dev = (x_max - x_min) / 10.0;
    let mut max = 0u16;
    let mut last_score = 0u32;
    let mut last_c = Complex::new((rng.gen::<f64>()-0.5)*4f64, (rng.gen::<f64>()-0.5)*4f64);
    let mut c;
    let mut num_try_outside_viewport: u32 = 0;
    'main_loop: for _ in 0..num_sample {
        
        if last_score == 0 {
            c = Complex::new((rng.gen::<f64>()-0.5)*4f64, (rng.gen::<f64>()-0.5)*4f64);
        } else {
            c = Complex::new(((rng.gen::<StandardNormal>().0 as f64 * std_dev) + last_c.re).min(2.0).max(-2.0), ((rng.gen::<StandardNormal>().0 as f64 * std_dev) + last_c.im).min(2.0).max(-2.0));
        }

        // check if c is in the cardioid || in the period-2 bulb
        let c_im_powi_2 = c.im.powi(2);
        let q = (c.re-0.25).powi(2)+ c_im_powi_2;
        if q*(q+(c.re-0.25)) < 0.25 * c_im_powi_2 || (c.re + 1.0).powi(2) + c_im_powi_2 < 0.0625{
            continue;
        }

        // find out if c is in the mandelbrot set
        let mut z = Complex::new(0f64,0f64);
        let mut i = 0u32;
        let mut escaped: bool = false;
        
        while i < std::cmp::min(max_iteration, 2000) {
            z = z*z + c;
            i += 1;
            if (z.re*z.re+z.im*z.im) >= 4f64 {escaped = true; break}
        }

        if i == max_iteration {continue}

        if !escaped {
            'fix_sized_outer_loop: while i <= max_iteration - 4000 {
                let old_z = z;
                for j in 0..4000 {
                    z = z*z + c;
                    if z == old_z {continue 'main_loop}
                    if (z.re*z.re+z.im*z.im) >= 4f64 {
                        escaped = true;
                        i += j;
                        break 'fix_sized_outer_loop
                    }
                }
                i += 4000;
            }
        }

        if i == max_iteration {continue}

        if !escaped {
            while i < max_iteration {
                z = z*z + c;
                i += 1;
            }
            if (z.re*z.re+z.im*z.im) >= 4f64 {escaped = true}
        }
        

        // if c escaped before max_iteration (therefore is NOT in the mandelbrot set)
        if escaped {
            // retrace the path of z
			z = c;
            let mut score = 0u32;
			while (z.re*z.re+z.im*z.im) < 4f64 {
				let coord_x = ((z.re-x_min)/(x_max-x_min)*size_x as f64) as i32;
				let coord_y = ((z.im-y_min)/(y_max-y_min)*size_y as f64) as i32;
				
                if coord_x >= 0 && coord_x < size_x as i32 {
                    if coord_y >= 0 && coord_y < size_y as i32 {
                        let new = image[coord_y as usize *size_x+coord_x as usize].saturating_add(1);
                        if new > max {
                            max = new;
                        }
                        if i > min_iteration {
                            image[coord_y as usize * size_x + coord_x as usize] = new;
                        }
                        score += 1;
                    }

                    // trace the symetric on axe x
                    let coord_y = ((-z.im-y_min)/(y_max-y_min)*size_y as f64) as i32;
                    if coord_y >= 0 && coord_y < size_y as i32 {
                        let new = image[coord_y as usize *size_x+coord_x as usize].saturating_add(1);
                        if new > max {
                            max = new;
                        }
                        if i > min_iteration {
                            image[coord_y as usize * size_x + coord_x as usize] = new;
                        }
                        score += 1;
                    }
                }
				
                z = z*z + c;			
            }

            if last_score == 0 {
                // searching for orbits crossing the viewport
                num_try_outside_viewport += 1;
            } else {
                // exploring an area containing orbits crossing the viewport
                num_try_outside_viewport -= 1;
            }

            if score >= last_score || score as f64 / last_score as f64 > rng.gen(){
                last_c = c;
                last_score = score;
            }
 
            if num_try_outside_viewport == 0 {
                last_score = 0;
            }
            

            
		}
    }

    return Ok(max);
}

#[inline(always)]
fn trace_complex32(z:Complex<f32>, i: u32, size_x: usize, size_y:usize, x_min: f32, x_max:f32, y_min: f32, y_max: f32, min_iteration: u32, max_iteration: u32, image: &mut [u16], nb_color: usize, color_offset: usize) -> u32 {
    let mut score = 0;
    let coord_x = ((z.re-x_min)/(x_max-x_min)*size_x as f32) as i32;
    let coord_y = ((z.im-y_min)/(y_max-y_min)*size_y as f32) as i32;
    
    if i <= max_iteration && coord_x >= 0 && coord_x < size_x as i32 {
        if coord_y >= 0 && coord_y < size_y as i32 {
            let new = image[(coord_y as usize *size_x+coord_x as usize)*nb_color+color_offset].saturating_add(1);
            if i > min_iteration {
                image[(coord_y as usize * size_x + coord_x as usize)*nb_color+color_offset] = new;
            }
            score += 1;

        }

        // trace the symetric on axe x
        let coord_y = ((-z.im-y_min)/(y_max-y_min)*size_y as f32) as i32;
        if coord_y >= 0 && coord_y < size_y as i32 {
            let new = image[(coord_y as usize *size_x+coord_x as usize)*nb_color+color_offset].saturating_add(1);
            if i > min_iteration {
                image[(coord_y as usize * size_x + coord_x as usize)*nb_color+color_offset] = new;
            }
            score += 1;
        }
    }

    return score;
}

#[inline(always)]
fn trace_complex64(z:Complex<f64>, i: u32, size_x: usize, size_y:usize, x_min: f64, x_max:f64, y_min: f64, y_max: f64, min_iteration: u32, max_iteration: u32, image: &mut [u16], nb_color: usize, color_offset: usize) -> u32 {
    let mut score = 0;
    let coord_x = ((z.re-x_min)/(x_max-x_min)*size_x as f64) as i32;
    let coord_y = ((z.im-y_min)/(y_max-y_min)*size_y as f64) as i32;
    
    if i <= max_iteration && coord_x >= 0 && coord_x < size_x as i32 {
        if coord_y >= 0 && coord_y < size_y as i32 {
            let new = image[(coord_y as usize *size_x+coord_x as usize)*nb_color+color_offset].saturating_add(1);
            if i > min_iteration {
                image[(coord_y as usize * size_x + coord_x as usize)*nb_color+color_offset] = new;
            }
            score += 1;

        }

        // trace the symetric on axe x
        let coord_y = ((-z.im-y_min)/(y_max-y_min)*size_y as f64) as i32;
        if coord_y >= 0 && coord_y < size_y as i32 {
            let new = image[(coord_y as usize *size_x+coord_x as usize)*nb_color+color_offset].saturating_add(1);
            if i > min_iteration {
                image[(coord_y as usize * size_x + coord_x as usize)*nb_color+color_offset] = new;
            }
            score += 1;
        }
    }

    return score;
}

pub fn gen_buddhabrot_color(size_x: usize, size_y:usize, x_min: f32, x_max:f32, y_min: f32, y_max: f32, red_min_iteration: u32, red_max_iteration: u32, green_min_iteration: u32, green_max_iteration: u32, blue_min_iteration: u32, blue_max_iteration: u32, num_sample: u32, image: &mut [u16]) -> Result<()> {
    let mut rng = rand::thread_rng();
    //let seed: &[_] = &[1, 2, 3, 4];
    //let mut rng: StdRng = SeedableRng::from_seed(seed);
    let max_iteration = red_max_iteration.max(green_max_iteration).max(blue_max_iteration);


    'main_loop: for _ in 0..num_sample {
        // generate a complex from a uniform rectangular distribution covering the mandelbrot set
        let c = Complex::new((rng.gen::<f32>()-0.5)*4f32, (rng.gen::<f32>()-0.5)*4f32);

        // check if c is in the cardioid || in the period-2 bulb
        let c_im_powi_2 = c.im.powi(2);
        let q = (c.re-0.25).powi(2)+ c_im_powi_2;
        if q*(q+(c.re-0.25)) < 0.25 * c_im_powi_2 || (c.re + 1.0).powi(2) + c_im_powi_2 < 0.0625{
            continue;
        }

        // find out if c is in the mandelbrot set
        let mut z = Complex::new(0f32,0f32);
        let mut i = 0u32;
        let mut escaped: bool = false;
        
        while i < std::cmp::min(max_iteration, 2000) {
            z = z*z + c;
            i += 1;
            if (z.re*z.re+z.im*z.im) >= 4f32 {escaped = true; break}
        }

        if i == max_iteration {continue}

        if !escaped {
            'fix_sized_outer_loop: while i <= max_iteration - 4000 {
                let old_z = z;
                for j in 0..4000 {
                    z = z*z + c;
                    if z == old_z {continue 'main_loop}
                    if (z.re*z.re+z.im*z.im) >= 4f32 {
                        escaped = true;
                        i += j;
                        break 'fix_sized_outer_loop
                    }
                }
                i += 4000;
            }
        }

        if i == max_iteration {continue}

        if !escaped {
            while i < max_iteration {
                z = z*z + c;
                i += 1;
            }
            if (z.re*z.re+z.im*z.im) >= 4f32 {escaped = true}
        }
        

        // if c escaped before max_iteration (therefore is NOT in the mandelbrot set)
        if escaped {
            // retrace the path of z
            z = c;
            while (z.re*z.re+z.im*z.im) < 4f32 {
                // red
                trace_complex32(z, i, size_x, size_y, x_min, x_max, y_min, y_max, red_min_iteration, red_max_iteration, image, 3, 0);
                // green
                trace_complex32(z, i, size_x, size_y, x_min, x_max, y_min, y_max, green_min_iteration, green_max_iteration, image, 3, 1);
                // blue
                trace_complex32(z, i, size_x, size_y, x_min, x_max, y_min, y_max, blue_min_iteration, blue_max_iteration, image, 3, 2);
                
                z = z*z + c;            
            }

        }
    }

    return Ok(());
    }


pub fn gen_buddhabrot_metropolis_color(size_x: usize, size_y:usize, x_min: f64, x_max:f64, y_min: f64, y_max: f64, red_min_iteration: u32, red_max_iteration: u32, green_min_iteration: u32, green_max_iteration: u32, blue_min_iteration: u32, blue_max_iteration: u32, num_sample: u32, image: &mut [u16]) -> Result<()> {
    let mut rng = rand::thread_rng();
    //let seed: &[_] = &[1, 2, 3, 4];
    //let mut rng: StdRng = SeedableRng::from_seed(seed);
    let max_iteration = red_max_iteration.max(green_max_iteration).max(blue_max_iteration);
    let std_dev = (x_max - x_min) / 10.0;
    let mut last_score = 0u32;
    let mut last_c = Complex::new((rng.gen::<f64>()-0.5)*4f64, (rng.gen::<f64>()-0.5)*4f64);
    let mut c;
    let mut num_try_outside_viewport: u32 = 0;
    'main_loop: for _ in 0..num_sample {
        
        if last_score == 0 {
            c = Complex::new((rng.gen::<f64>()-0.5)*4f64, (rng.gen::<f64>()-0.5)*4f64);
        } else {
            c = Complex::new(((rng.gen::<StandardNormal>().0 as f64 * std_dev) + last_c.re).min(2.0).max(-2.0), ((rng.gen::<StandardNormal>().0 as f64 * std_dev) + last_c.im).min(2.0).max(-2.0));
        }

        // check if c is in the cardioid || in the period-2 bulb
        let c_im_powi_2 = c.im.powi(2);
        let q = (c.re-0.25).powi(2)+ c_im_powi_2;
        if q*(q+(c.re-0.25)) < 0.25 * c_im_powi_2 || (c.re + 1.0).powi(2) + c_im_powi_2 < 0.0625{
            continue;
        }

        // find out if c is in the mandelbrot set
        let mut z = Complex::new(0f64,0f64);
        let mut i = 0u32;
        let mut escaped: bool = false;
        
        while i < std::cmp::min(max_iteration, 2000) {
            z = z*z + c;
            i += 1;
            if (z.re*z.re+z.im*z.im) >= 4f64 {escaped = true; break}
        }

        if i == max_iteration {continue}

        if !escaped {
            'fix_sized_outer_loop: while i <= max_iteration - 4000 {
                let old_z = z;
                for j in 0..4000 {
                    z = z*z + c;
                    if z == old_z {continue 'main_loop}
                    if (z.re*z.re+z.im*z.im) >= 4f64 {
                        escaped = true;
                        i += j;
                        break 'fix_sized_outer_loop
                    }
                }
                i += 4000;
            }
        }

        if i == max_iteration {continue}

        if !escaped {
            while i < max_iteration {
                z = z*z + c;
                i += 1;
            }
            if (z.re*z.re+z.im*z.im) >= 4f64 {escaped = true}
        }
        

        // if c escaped before max_iteration (therefore is NOT in the mandelbrot set)
        if escaped {
            // retrace the path of z
            z = c;
            let mut score = 0u32;
            while (z.re*z.re+z.im*z.im) < 4f64 {
                // red
                score += trace_complex64(z, i, size_x, size_y, x_min, x_max, y_min, y_max, red_min_iteration, red_max_iteration, image, 3, 0);
                // green
                score += trace_complex64(z, i, size_x, size_y, x_min, x_max, y_min, y_max, green_min_iteration, green_max_iteration, image, 3, 1);
                // blue
                score += trace_complex64(z, i, size_x, size_y, x_min, x_max, y_min, y_max, blue_min_iteration, blue_max_iteration, image, 3, 2);
                
                z = z*z + c;            
            }

            if last_score == 0 {
                // searching for orbits crossing the viewport
                num_try_outside_viewport += 1;
            } else {
                // exploring an area containing orbits crossing the viewport
                num_try_outside_viewport -= 1;
            }

            if score >= last_score || score as f64 / last_score as f64 > rng.gen(){
                last_c = c;
                last_score = score;
            }
 
            if num_try_outside_viewport == 0 {
                last_score = 0;
            }
            

            
        }
    }

    return Ok(());
}

