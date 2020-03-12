#ifndef CUSTOMMATERIAL_HPP
#define CUSTOMMATERIAL_HPP

#include <Teuchos_ParameterList.hpp>

#include <cstdarg>

namespace Plato
{

/******************************************************************************/
/*!
  \brief Class for custom material models
*/
class CustomMaterial
/******************************************************************************/
{
public:
    CustomMaterial(const Teuchos::ParameterList& aParamList) {}
    virtual ~CustomMaterial() {}

protected:
    virtual double GetCustomExpressionValue(
        const Teuchos::ParameterList& paramList,
        const std::string name ) const;

    virtual double GetCustomExpressionValue(
        const Teuchos::ParameterList& paramList,
        int equationIndex = -1,
        std::string equationName = std::string("Equation") ) const;
};
// class CustomMaterial

} // namespace Plato

#endif
