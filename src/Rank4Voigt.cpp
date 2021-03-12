/*
 * Rank4Voigt.cpp
 *
 *  Created on: Jul 11, 2020
 */

#include "MaterialModel.hpp"

namespace Plato {

  template<>
  Rank4VoigtFunctor<1>::Rank4VoigtFunctor(Teuchos::ParameterList& aParams) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011"); /*throw if not found*/
          c1[0][0] = Plato::ParseTools::getParam<T>(aParams, "c111", 0.0);
          c2[0][0] = Plato::ParseTools::getParam<T>(aParams, "c211", 0.0);
      }

  template<>
  Rank4VoigtFunctor<2>::Rank4VoigtFunctor(Teuchos::ParameterList& aParams) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011"); /*throw if not found*/
          c1[0][0] = Plato::ParseTools::getParam<T>(aParams, "c111", 0.0);
          c2[0][0] = Plato::ParseTools::getParam<T>(aParams, "c211", 0.0);

          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c022", /*default=*/ c0[0][0]);
          c1[1][1] = Plato::ParseTools::getParam<T>(aParams, "c122", /*default=*/ c1[0][0]);
          c2[1][1] = Plato::ParseTools::getParam<T>(aParams, "c222", /*default=*/ c2[0][0]);

          c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c033"); /*throw if not found*/
          c1[2][2] = Plato::ParseTools::getParam<T>(aParams, "c133", 0.0);
          c2[2][2] = Plato::ParseTools::getParam<T>(aParams, "c233", 0.0);

          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c012", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c1[0][1] = Plato::ParseTools::getParam<T>(aParams, "c112", /*default=*/ 0.0); c1[1][0] = c1[0][1];
          c2[0][1] = Plato::ParseTools::getParam<T>(aParams, "c212", /*default=*/ 0.0); c2[1][0] = c2[0][1];
      }

  template<>
  Rank4VoigtFunctor<3>::Rank4VoigtFunctor(Teuchos::ParameterList& aParams) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);
          c1[0][0] = Plato::ParseTools::getParam<T>(aParams, "c111", 0.0);
          c2[0][0] = Plato::ParseTools::getParam<T>(aParams, "c211", 0.0);

          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c022", /*default=*/ c0[0][0]);
          c1[1][1] = Plato::ParseTools::getParam<T>(aParams, "c122", /*default=*/ c1[0][0]);
          c2[1][1] = Plato::ParseTools::getParam<T>(aParams, "c222", /*default=*/ c2[0][0]);

          c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c033", /*default=*/ c0[0][0]);
          c1[2][2] = Plato::ParseTools::getParam<T>(aParams, "c133", /*default=*/ c1[0][0]);
          c2[2][2] = Plato::ParseTools::getParam<T>(aParams, "c233", /*default=*/ c2[0][0]);

          c0[3][3] = Plato::ParseTools::getParam<T>(aParams, "c044" /*throw if not found*/);
          c1[3][3] = Plato::ParseTools::getParam<T>(aParams, "c144", 0.0);
          c2[3][3] = Plato::ParseTools::getParam<T>(aParams, "c244", 0.0);

          c0[4][4] = Plato::ParseTools::getParam<T>(aParams, "c055", /*default=*/ c0[3][3]);
          c1[4][4] = Plato::ParseTools::getParam<T>(aParams, "c155", /*default=*/ c1[3][3]);
          c2[4][4] = Plato::ParseTools::getParam<T>(aParams, "c255", /*default=*/ c2[3][3]);

          c0[5][5] = Plato::ParseTools::getParam<T>(aParams, "c066", /*default=*/ c0[3][3]);
          c1[5][5] = Plato::ParseTools::getParam<T>(aParams, "c166", /*default=*/ c1[3][3]);
          c2[5][5] = Plato::ParseTools::getParam<T>(aParams, "c266", /*default=*/ c2[3][3]);

          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c012", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c1[0][1] = Plato::ParseTools::getParam<T>(aParams, "c112", /*default=*/ 0.0); c1[1][0] = c1[0][1];
          c2[0][1] = Plato::ParseTools::getParam<T>(aParams, "c212", /*default=*/ 0.0); c2[1][0] = c2[0][1];

          c0[0][2] = Plato::ParseTools::getParam<T>(aParams, "c013", /*default=*/ c0[0][1]); c0[2][0] = c0[0][2];
          c1[0][2] = Plato::ParseTools::getParam<T>(aParams, "c113", /*default=*/ c1[0][1]); c1[2][0] = c1[0][2];
          c2[0][2] = Plato::ParseTools::getParam<T>(aParams, "c213", /*default=*/ c2[0][1]); c2[2][0] = c2[0][2];

          c0[1][2] = Plato::ParseTools::getParam<T>(aParams, "c023", /*default=*/ c0[0][1]); c0[2][1] = c0[1][2];
          c1[1][2] = Plato::ParseTools::getParam<T>(aParams, "c123", /*default=*/ c1[0][1]); c1[2][1] = c1[1][2];
          c2[1][2] = Plato::ParseTools::getParam<T>(aParams, "c223", /*default=*/ c2[0][1]); c2[2][1] = c2[1][2];
      }

  template<>
  Rank4VoigtConstant<1>::Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c11" /*throw if not found*/);

      }

  template<>
  Rank4VoigtConstant<2>::Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c11" /*throw if not found*/);
          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c22", /*default=*/ c0[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c33" /*throw if not found*/);
      }

  template<>
  Rank4VoigtConstant<3>::Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c11" /*throw if not found*/);
          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c22", /*default=*/ c0[0][0]);
          c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c33", /*default=*/ c0[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c0[0][2] = Plato::ParseTools::getParam<T>(aParams, "c13", /*default=*/ c0[0][1]); c0[2][0] = c0[0][2];
          c0[1][2] = Plato::ParseTools::getParam<T>(aParams, "c23", /*default=*/ c0[0][1]); c0[2][1] = c0[1][2];
          c0[3][3] = Plato::ParseTools::getParam<T>(aParams, "c44" /*throw if not found*/);
          c0[4][4] = Plato::ParseTools::getParam<T>(aParams, "c55", c0[3][3]);
          c0[5][5] = Plato::ParseTools::getParam<T>(aParams, "c66", c0[3][3]);
      }

} // namespace Plato
