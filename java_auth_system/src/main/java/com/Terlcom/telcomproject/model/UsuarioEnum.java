package com.Terlcom.telcomproject.model;

public enum UsuarioEnum {
    ADMIN("Admim"),
    USER("User");



    String role;
    UsuarioEnum(String role) {
        this.role = role;
    }

    public String getRole() {
        return role;
    }
}
