set -e

# Script that tries to "test" all notebooks by at least making sure they run.
# This is a bit tedious but tries to massage each notebook so it can be at least run and has no obvious errors.

jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_first_derivative_solutions.ipynb" | grep -v ipython | sed "s/plt\.show/plt\.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/ac1d_optimal_operator.ipynb" | grep -v nbagg | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/ac1d_optimal_operator_with_solutions.ipynb" | grep -v nbagg | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac1d.ipynb" | grep -v nbagg | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac1d_with_solutions.ipynb" | grep -v nbagg | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac2d_heterogeneous.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac2d_heterogeneous_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show(.*)/plt.close()/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac2d_homogeneous.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac2d_homogeneous_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac3d_homogeneous.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_ac3d_homogeneous_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_advection_1d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_advection_1d_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_advection_diffusion_reaction.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_advection_diffusion_reaction_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_elastic1d_staggered.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_elastic1d_staggered_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_first_derivative.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | sed "s/#ffder\[it\]=/pass/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_first_derivative_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_seismometer.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | sed "s/^fu0 = float.*$/fu0 = 1.0/g" | sed "s/h = float.*$/h = 0.5/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_seismometer_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | sed "s/^fu0 = float.*$/fu0 = 1.0/g" | sed "s/h = float.*$/h = 0.5/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_taylor_operators.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Difference Method/fd_taylor_operators_advanced.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

cd "notebooks/Ambient Seismic Noise/"
jupyter-nbconvert --to script --stdout "Probabilistic Power Spectral Densities.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | sed 's/plot(/plot(show=False,/g' | python
cd ../..

cd "notebooks/Computational Seismology/The Discontinuous Galerkin Method"
jupyter-nbconvert --to script --stdout "dg_elastic_hetero_1d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
## We only test the solution notebook here.
jupyter-nbconvert --to script --stdout "dg_elastic_homo_1d_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "dg_scalar_advection_1d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
cd ../../..

# Only test solution
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Element Method/fe_elastic_1d_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Element Method/fe_static_elasticity.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Volume Method/fv_elastic_hetero_1d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

# Only test solution.
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Volume Method/fv_elastic_homo_1d_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Finite-Volume Method/fv_scalar_advection_1d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Pseudospectral Method/ps_cheby_derivative.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/The Pseudospectral Method/ps_cheby_derivative_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

cd "notebooks/Computational Seismology/The Pseudospectral Method"
# Only test solution.
jupyter-nbconvert --to script --stdout "ps_cheby_elastic_1d_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "ps_fourier_acoustic_1d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "ps_fourier_acoustic_2d.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "ps_fourier_derivative.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "ps_fourier_derivative_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
cd ../../..

cd "notebooks/Computational Seismology/The Spectral-Element Method"
# Only test solution.
jupyter-nbconvert --to script --stdout "se_hetero_1d_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
# Only test solution.
jupyter-nbconvert --to script --stdout "se_homo_1d_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "se_Lagrange_interpolation.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | sed "s/N = int.*/N = 4/g" | python
jupyter-nbconvert --to script --stdout "se_numerical_integration_GLL.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | sed "s/N =int.*/N = 4/g" | python
cd ../../..

jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/Wave Propagation & Analytical Solutions/Greens_function_acoustic_1-3D.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/Wave Propagation & Analytical Solutions/time_reversal_reciprocity.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/Wave Propagation & Analytical Solutions/Double_couple_homogeneous_3D/Double_couple_homogeneous_3D.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

# The following two require a fortran script and it escapes to bash. need to change that.
#jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/Wave Propagation & Analytical Solutions/Lambs_problem_3D/lambs_problem.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
#jupyter-nbconvert --to script --stdout "notebooks/Computational Seismology/Wave Propagation & Analytical Solutions/Lambs_problem_3D/lambs_problem_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

# This is too interactive to be tested in this manner..
#jupyter-nbconvert --to script --stdout "notebooks/Earthquake Physics/rsf_widgets_dashboard.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python

exit
# These are not yet done...
jupyter-nbconvert --to script --stdout "notebooks/General Seismology/instrument_response.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_01_introduction.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_02_basis.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_02_basis_with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_03_record_section.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_03_record_section_with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_04_finite_source.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/Instaseis_Tutorial_04_finite_source_with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Instaseis-Syngine/syngine_tutorial.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/00_Introduction.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/01_File_Formats-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/01_File_Formats.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/02_UTCDateTime-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/02_UTCDateTime.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/03_waveform_data-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/03_waveform_data.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/04_Station_metainformation-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/04_Station_metainformation.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/05_Event_metadata-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/05_Event_metadata.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/06_FDSN-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/06_FDSN.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/07_Basic_Processing_Exercise-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/07_Basic_Processing_Exercise.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/08_Exercise__2008_MtCarmel_Earthquake_and_Aftershock_Series-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/ObsPy/08_Exercise__2008_MtCarmel_Earthquake_and_Aftershock_Series.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Python Introduction/Python_Crash_Course-with_solutions.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Python Introduction/Python_Crash_Course.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_1_phase_relative_times.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_2_source_width.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_3_finite_source_seismograms.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_4_earth_models.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_5_compare_seismograms_for_models.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_6_data_quality.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Reproducible Papers/Syngine_2016/figure_8_education.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Rotational Seismology/download+preprocess_data.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Rotational Seismology/estimate_backazimuth.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Rotational Seismology/estimate_phase_velocity.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Signal Processing/filter_basics.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Signal Processing/filter_basics_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Signal Processing/fourier_transform.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Signal Processing/fourier_transform_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Signal Processing/spectral_analysis+preprocessing.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
jupyter-nbconvert --to script --stdout "notebooks/Signal Processing/spectral_analysis+preprocessing_solution.ipynb" | grep -v nbagg | grep -v ipython | sed "s/plt.show/plt.close/g" | python
