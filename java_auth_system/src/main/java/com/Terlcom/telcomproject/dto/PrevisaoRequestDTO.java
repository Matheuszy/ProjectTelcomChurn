package com.Terlcom.telcomproject.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PrevisaoRequestDTO(

        @JsonProperty("CidadaoSenior")
        Integer cidadaoSenior,

        @JsonProperty("Parceiro")
        Integer parceiro,

        @JsonProperty("Dependentes")
        Integer dependentes,

        @JsonProperty("Fidelidade")
        Integer fidelidade,

        @JsonProperty("ServicoTelefonico")
        Integer servicoTelefonico,

        @JsonProperty("SuporteTecnico")
        Integer suporteTecnico,

        @JsonProperty("StreamingTV")
        Integer streamingTV,

        @JsonProperty("MensalCobrado")
        Float mensalCobrado,

        @JsonProperty("TotalCobrado")
        Float totalCobrado,

        @JsonProperty("MultiplasLinhas_No")
        Integer multiplasLinhasNo,

        @JsonProperty("MultiplasLinhas_No_phone_service")
        Integer multiplasLinhasNoPhoneService,

        @JsonProperty("MultiplasLinhas_Yes")
        Integer multiplasLinhasYes,

        @JsonProperty("ServicoInternet_DSL")
        Integer servicoInternetDSL,

        @JsonProperty("ServicoInternet_Fiber_optic")
        Integer servicoInternetFiberOptic,

        @JsonProperty("ServicoInternet_No")
        Integer servicoInternetNo,

        @JsonProperty("Contrato_Month_to_month")
        Integer contratoMonthToMonth,

        @JsonProperty("Contrato_One_year")
        Integer contratoOneYear,

        @JsonProperty("Contrato_Two_year")
        Integer contratoTwoYear
) {}
