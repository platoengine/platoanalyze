#include <sstream>

#include "AmgXConfigs.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{
  std::string configurationString(std::string aConfigOption, Plato::Scalar aTolerance, int aMaxIters, bool aAbsTolType)
  {
      using namespace std;
      std::ostringstream tStrStream;
      if(aConfigOption == "eaf")
      {
          tStrStream <<
               "{\
                   \"config_version\": 2,\
                   \"solver\": {\
                       \"preconditioner\": {\
                           \"print_grid_stats\": 1,\
                           \"print_vis_data\": 1,\
                           \"solver\": \"AMG\",\
                           \"algorithm\": \"AGGREGATION\",\
                           \"max_levels\": 50,\
                           \"dense_lu_num_rows\":10000,\
                           \"selector\": \"SIZE_8\",\
                           \"smoother\": {\
                               \"scope\": \"jacobi\",\
                               \"solver\": \"BLOCK_JACOBI\",\
                               \"monitor_residual\": 1,\
                               \"print_solve_stats\": 1\
                           },\
                           \"print_solve_stats\": 1,\
                           \"presweeps\": 1,\
                           \"max_iters\": 1,\
                           \"monitor_residual\": 1,\
                           \"store_res_history\": 0,\
                           \"scope\": \"amg\",\
                           \"cycle\": \"CGF\",\
                           \"postsweeps\": 1\
                       },\
                       \"solver\": \"PCG\",\
                       \"print_solve_stats\": 1,\
                       \"print_config\": 0,\
                       \"obtain_timings\": 1,\
                       \"monitor_residual\": 1,\
                       \"convergence\": \"";
                       if (aAbsTolType == false)
                         tStrStream << "RELATIVE_INI_CORE";
                       else if (aAbsTolType == true)
                         tStrStream << "ABSOLUTE";
                       tStrStream << "\", \
                       \"scope\": \"main\",\
                       \"tolerance\": " << aTolerance << ",\
                       \"max_iters\": " << aMaxIters << ",\
                       \"norm\": \"L2\"\
                   }\
               }";
      } else
      if(aConfigOption == "default")
      {
          tStrStream <<
             "{\
                \"config_version\": 2,\
                \"solver\": {\
                  \"preconditioner\": {\
                      \"print_grid_stats\": 1,\
                      \"algorithm\": \"AGGREGATION\",\
                      \"print_vis_data\": 0,\
                      \"max_matching_iterations\": 50,\
                      \"max_unassigned_percentage\": 0.01,\
                      \"solver\": \"AMG\",\
                      \"smoother\": {\
                          \"relaxation_factor\": 0.78,\
                          \"scope\": \"jacobi\",\
                          \"solver\": \"BLOCK_JACOBI\",\
                          \"monitor_residual\": 0,\
                          \"print_solve_stats\": 0\
                      },\
                      \"print_solve_stats\": 0,\
                      \"dense_lu_num_rows\": 64,\
                      \"presweeps\": 1,\
                      \"selector\": \"SIZE_8\",\
                      \"coarse_solver\": \"DENSE_LU_SOLVER\",\
                      \"coarsest_sweeps\": 2,\
                      \"max_iters\": 1,\
                      \"monitor_residual\": 0,\
                      \"store_res_history\": 0,\
                      \"scope\": \"amg\",\
                      \"max_levels\": 100,\
                      \"postsweeps\": 1,\
                      \"cycle\": \"W\"\
                  },\
                  \"solver\": \"PBICGSTAB\",\
                  \"print_solve_stats\": 0,\
                  \"obtain_timings\": 0,\
                  \"max_iters\": 1000,\
                  \"monitor_residual\": 1,\
                  \"convergence\": \"";
                  if (aAbsTolType == false)
                    tStrStream << "RELATIVE_INI_CORE";
                  else if (aAbsTolType == true)
                    tStrStream << "ABSOLUTE";
                  tStrStream << "\", \
                  \"scope\": \"main\",\
                  \"tolerance\": 5.0e-08,\
                  \"norm\": \"L2\"\
              }\
          }";
      } else
      if(aConfigOption == "pcg_noprec")
      {
            tStrStream <<
               "{\
                   \"config_version\": 2, \
                   \"solver\": {\
                       \"preconditioner\": {\
                           \"scope\": \"amg\", \
                           \"solver\": \"NOSOLVER\"\
                       }, \
                       \"use_scalar_norm\": 1, \
                       \"solver\": \"PCG\", \
                       \"print_solve_stats\": 1, \
                       \"obtain_timings\": 1, \
                       \"monitor_residual\": 1, \
                       \"convergence\": \"";
                       if (aAbsTolType == false)
                         tStrStream << "RELATIVE_INI_CORE";
                       else if (aAbsTolType == true)
                         tStrStream << "ABSOLUTE";
                       tStrStream << "\", \
                       \"scope\": \"main\", \
                       \"tolerance\": " << aTolerance << ", \
                       \"max_iters\": " << aMaxIters << ", \
                       \"norm\": \"L2\"\
                   }\
               }";
      } else
      if(aConfigOption == "pcg_v")
      {
            tStrStream <<
               "{\
                   \"config_version\": 2, \
                   \"solver\": {\
                       \"preconditioner\": {\
                           \"print_grid_stats\": 1, \
                           \"print_vis_data\": 0, \
                           \"solver\": \"AMG\", \
                           \"smoother\": {\
                               \"scope\": \"jacobi\", \
                               \"solver\": \"BLOCK_JACOBI\", \
                               \"monitor_residual\": 0, \
                               \"print_solve_stats\": 0\
                           }, \
                           \"print_solve_stats\": 0, \
                           \"presweeps\": 1, \
                           \"max_iters\": 1, \
                           \"monitor_residual\": 0, \
                           \"store_res_history\": 0, \
                           \"scope\": \"amg\", \
                           \"max_levels\": 100, \
                           \"cycle\": \"V\", \
                           \"postsweeps\": 1\
                       }, \
                       \"solver\": \"PCG\", \
                       \"print_solve_stats\": 1, \
                       \"obtain_timings\": 1, \
                       \"max_iters\": " << aMaxIters << ", \
                       \"monitor_residual\": 1, \
                       \"convergence\": \"";
                       if (aAbsTolType == false)
                         tStrStream << "RELATIVE_INI_CORE";
                       else if (aAbsTolType == true)
                         tStrStream << "ABSOLUTE";
                       tStrStream << "\", \
                       \"scope\": \"main\", \
                       \"tolerance\": " << aTolerance << ", \
                       \"norm\": \"L2\"\
                   }\
               }";
      } else
      if(aConfigOption == "pcg_w")
      {
            tStrStream <<
               "{\
                   \"config_version\": 2, \
                   \"solver\": {\
                       \"preconditioner\": {\
                           \"print_grid_stats\": 1, \
                           \"print_vis_data\": 0, \
                           \"solver\": \"AMG\", \
                           \"smoother\": {\
                               \"scope\": \"jacobi\", \
                               \"solver\": \"BLOCK_JACOBI\", \
                               \"monitor_residual\": 0, \
                               \"print_solve_stats\": 0\
                           }, \
                           \"print_solve_stats\": 0, \
                           \"presweeps\": 1, \
                           \"max_iters\": 1, \
                           \"monitor_residual\": 0, \
                           \"store_res_history\": 0, \
                           \"scope\": \"amg\", \
                           \"max_levels\": 100, \
                           \"cycle\": \"W\", \
                           \"postsweeps\": 1\
                       }, \
                       \"solver\": \"PCG\", \
                       \"print_solve_stats\": 1, \
                       \"obtain_timings\": 1, \
                       \"max_iters\": " << aMaxIters << ", \
                       \"monitor_residual\": 1, \
                       \"convergence\": \"";
                       if (aAbsTolType == false)
                         tStrStream << "RELATIVE_INI_CORE";
                       else if (aAbsTolType == true)
                         tStrStream << "ABSOLUTE";
                       tStrStream << "\", \
                       \"scope\": \"main\", \
                       \"tolerance\": " << aTolerance << ", \
                       \"norm\": \"L2\"\
                   }\
               }";
      } else
      if(aConfigOption == "pcg_f")
      {
            tStrStream <<
               "{\
                   \"config_version\": 2, \
                   \"solver\": {\
                       \"preconditioner\": {\
                           \"print_grid_stats\": 1, \
                           \"print_vis_data\": 0, \
                           \"solver\": \"AMG\", \
                           \"smoother\": {\
                               \"scope\": \"jacobi\", \
                               \"solver\": \"BLOCK_JACOBI\", \
                               \"monitor_residual\": 0, \
                               \"print_solve_stats\": 0\
                           }, \
                           \"print_solve_stats\": 0, \
                           \"presweeps\": 1, \
                           \"max_iters\": 1, \
                           \"monitor_residual\": 0, \
                           \"store_res_history\": 0, \
                           \"scope\": \"amg\", \
                           \"max_levels\": 100, \
                           \"cycle\": \"F\", \
                           \"postsweeps\": 1\
                       }, \
                       \"solver\": \"PCG\", \
                       \"print_solve_stats\": 1, \
                       \"obtain_timings\": 1, \
                       \"max_iters\": " << aMaxIters << ", \
                       \"monitor_residual\": 1, \
                       \"convergence\": \"";
                       if (aAbsTolType == false)
                         tStrStream << "RELATIVE_INI_CORE";
                       else if (aAbsTolType == true)
                         tStrStream << "ABSOLUTE";
                       tStrStream << "\", \
                       \"scope\": \"main\", \
                       \"tolerance\": " << aTolerance << ", \
                       \"norm\": \"L2\"\
                   }\
               }";
      } else
      if(aConfigOption == "agg_cheb4")
      {
 	    tStrStream <<
            "{\
                \"config_version\": 2, \
                \"determinism_flag\": 1, \
                \"solver\": {\
                    \"print_grid_stats\": 1, \
                    \"algorithm\": \"AGGREGATION\", \
                    \"obtain_timings\": 1, \
                    \"error_scaling\": 3,\
                    \"solver\": \"AMG\", \
                    \"smoother\": \
                    {\
                        \"solver\": \"CHEBYSHEV\",\
                        \"preconditioner\" : \
                        {\
                            \"solver\": \"JACOBI_L1\",\
                            \"max_iters\": 1\
                        },\
                        \"max_iters\": 1,\
                        \"chebyshev_polynomial_order\" : 4,\
                        \"chebyshev_lambda_estimate_mode\" : 2\
                    },\
                    \"presweeps\": 0, \
                    \"postsweeps\": 1, \
                    \"print_solve_stats\": 1, \
                    \"selector\": \"SIZE_8\", \
                    \"coarsest_sweeps\": 1, \
                    \"monitor_residual\": 1, \
                    \"min_coarse_rows\": 2, \
                    \"scope\": \"main\", \
                    \"max_levels\": 1000, \
                    \"convergence\": \"";
                    if (aAbsTolType == false)
                      tStrStream << "RELATIVE_INI_CORE";
                    else if (aAbsTolType == true)
                      tStrStream << "ABSOLUTE";
                    tStrStream << "\", \
                    \"tolerance\": " << aTolerance << ",\
                    \"max_iters\": " << aMaxIters << ",\
                    \"norm\": \"L2\",\
                    \"cycle\": \"V\"\
                }\
            }\
            ";
      }
      else
      {
          THROWERR("AMGX configuration string is not predefined.")
      }
      return tStrStream.str();
  }

}
