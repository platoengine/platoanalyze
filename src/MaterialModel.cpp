#include "MaterialModel.hpp"

namespace Plato {

  template<>
  TensorFunctor<1>::TensorFunctor(Teuchos::ParameterList& aParams) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);
          c1[0][0] = Plato::ParseTools::getParam<T>(aParams, "c111", /*default=*/ 0.0);
          c2[0][0] = Plato::ParseTools::getParam<T>(aParams, "c211", /*default=*/ 0.0);
      }

  template<>
  TensorFunctor<2>::TensorFunctor(Teuchos::ParameterList& aParams) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);
          c1[0][0] = Plato::ParseTools::getParam<T>(aParams, "c111", /*default=*/ 0.0);
          c2[0][0] = Plato::ParseTools::getParam<T>(aParams, "c211", /*default=*/ 0.0);

          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c022", /*default=*/ c0[0][0]);
          c1[1][1] = Plato::ParseTools::getParam<T>(aParams, "c122", /*default=*/ c1[0][0]);
          c2[1][1] = Plato::ParseTools::getParam<T>(aParams, "c222", /*default=*/ c2[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c012", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c1[0][1] = Plato::ParseTools::getParam<T>(aParams, "c112", /*default=*/ 0.0); c1[1][0] = c1[0][1];
          c2[0][1] = Plato::ParseTools::getParam<T>(aParams, "c212", /*default=*/ 0.0); c2[1][0] = c2[0][1];
      }

  template<>
  TensorFunctor<3>::TensorFunctor(Teuchos::ParameterList& aParams) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);
          c1[0][0] = Plato::ParseTools::getParam<T>(aParams, "c111", /*default=*/ 0.0);
          c2[0][0] = Plato::ParseTools::getParam<T>(aParams, "c211", /*default=*/ 0.0);

          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c022", /*default=*/ c0[0][0]);
          c1[1][1] = Plato::ParseTools::getParam<T>(aParams, "c122", /*default=*/ c1[0][0]);
          c2[1][1] = Plato::ParseTools::getParam<T>(aParams, "c222", /*default=*/ c2[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c012", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c1[0][1] = Plato::ParseTools::getParam<T>(aParams, "c112", /*default=*/ 0.0); c1[1][0] = c1[0][1];
          c2[0][1] = Plato::ParseTools::getParam<T>(aParams, "c212", /*default=*/ 0.0); c2[1][0] = c2[0][1];

          c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c033", /*default=*/ c0[0][0]);
          c1[2][2] = Plato::ParseTools::getParam<T>(aParams, "c133", /*default=*/ c1[0][0]);
          c2[2][2] = Plato::ParseTools::getParam<T>(aParams, "c233", /*default=*/ c2[0][0]);
          c0[0][2] = Plato::ParseTools::getParam<T>(aParams, "c013", /*default=*/ 0.0); c0[2][0] = c0[0][2];
          c1[0][2] = Plato::ParseTools::getParam<T>(aParams, "c113", /*default=*/ 0.0); c1[2][0] = c1[0][2];
          c2[0][2] = Plato::ParseTools::getParam<T>(aParams, "c213", /*default=*/ 0.0); c2[2][0] = c2[0][2];
          c0[1][2] = Plato::ParseTools::getParam<T>(aParams, "c023", /*default=*/ 0.0); c0[2][1] = c0[1][2];
          c1[1][2] = Plato::ParseTools::getParam<T>(aParams, "c123", /*default=*/ 0.0); c1[2][1] = c1[1][2];
          c2[1][2] = Plato::ParseTools::getParam<T>(aParams, "c223", /*default=*/ 0.0); c2[2][1] = c2[1][2];
      }

  template<>
  TensorConstant<1>::TensorConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);

      }

  template<>
  TensorConstant<2>::TensorConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);
          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c022", /*default=*/ c0[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c012", /*default=*/ 0.0); c0[1][0] = c0[0][1];
      }

  template<>
  TensorConstant<3>::TensorConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
      {
          typedef Plato::Scalar T;
          c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c011" /*throw if not found*/);
          c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c022", /*default=*/ c0[0][0]);
          c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c012", /*default=*/ 0.0); c0[1][0] = c0[0][1];
          c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c033", /*default=*/ c0[0][0]);
          c0[0][2] = Plato::ParseTools::getParam<T>(aParams, "c013", /*default=*/ 0.0); c0[2][0] = c0[0][2];
          c0[1][2] = Plato::ParseTools::getParam<T>(aParams, "c023", /*default=*/ 0.0); c0[2][1] = c0[1][2];
      }

} // namespace Plato
