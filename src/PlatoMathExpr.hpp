/*
//@HEADER
// *************************************************************************
//   Plato Engine v.1.0: Copyright 2018, National Technology & Engineering
//                    Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Sandia Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact the Plato team (plato3D-help@sandia.gov)
//
// *************************************************************************
//@HEADER
*/

/*
 * Plato_MathExpr.hpp
 *
 *  Created on: Aug 4, 2020
 */

#pragma once
#include <Teuchos_MathExpr.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Basic math expression evaluator
 *  This evaluator executes exclusively on the host.
 *  The function is of the form y = f(t) where y and t are doubles.
 *  The expression, f(t), is provided as a string argument to the constructor and
 *  parameters can be defined in the string argument, e.g., "pi=3.14159264; sin(pi*t)".
**********************************************************************************/
class MathExpr
{
  public:

    /******************************************************************************//**
     * @brief Constructor
    **********************************************************************************/
    explicit MathExpr(std::string aExpression) : mExpression(aExpression)
    {
        mReader = Teuchos::rcp(Teuchos::MathExpr::new_calc_reader());
    }

    /******************************************************************************//**
     * @brief Return f(aInVal)
     * @param aInVal input value
     * @return f(aInVal)
    **********************************************************************************/
    inline Plato::Scalar
    value(Plato::Scalar aInVal) const
    {
        Teuchos::any result_any;
        std::stringstream ss;
        ss << "t=" << std::fixed << std::setprecision(18) << aInVal << "; " << mExpression;
        mReader->read_string(result_any, ss.str(), "expression");
        auto result = Teuchos::any_cast<Plato::Scalar>(result_any);
        return result;
    }

  private:

    Teuchos::RCP<Teuchos::Reader> mReader;
    std::string mExpression;

}; // class MathExpr

} // namespace Plato
