# 
# Copyright (C) 2020
#
# This file is a part of PROPCOT / OPKA
# 
# BNEvidenceBase library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301  USA
#



import earnedvaluemanagement as evm


inputName = "ModelSettings_Example_2.xlsx"

PRJ = evm.project_reader(inputName)
case = evm.build_model(PRJ)
case_sample = evm.sample_model(case)
#evm.display_posterior(case_sample,inputName)
evm.excel_posterior(case_sample,inputName)

