"""
masks.py - Gerador de máscaras de fase e intensidade para feixes Laguerre-Gauss
utilizando a biblioteca diffractio.

Baseado na geração de modos Laguerre-Gauss (n, l), o script produz:
1. Máscaras de fase para SLM (uint8 [0, 255] de 0 a 2*pi, resolução 1:1, sem bordas/eixos).
2. Máscaras com rede de difração (forked grating / blazed grating) opcionais para SLM.
3. Imagens PNG individuais de fase (escala de cinza).
4. Imagens PNG individuais de intensidade (|u|² com colormap gist_heat).
5. Painéis gerais comparativos de amplitude/intensidade e fase.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from diffractio import mm, um
from diffractio.scalar_sources_XY import Scalar_source_XY


def gerar_mascaras(
    N: int = 0,
    K: int = 3,
    num_pixels: int = 512,
    tamanho_janela_mm: float = 2.0,
    wavelength_um: float = 0.6238,
    w0_um: float = 250.0,
    periodo_rede_px: float = 0.0,
    pasta_saida: str = "mascaras_saida",
    salvar_individuais: bool = True,
    salvar_grids: bool = True,
):
    """
    Gera máscaras de fase e intensidade para feixes Laguerre-Gauss.

    Parâmetros:
    -----------
    N : int
        Índice radial máximo n (de 0 a N).
    K : int
        Carga topológica / índice azimutal máximo l (de 0 a K).
    num_pixels : int
        Número de amostras nos eixos x e y (ex: 512).
    tamanho_janela_mm : float
        Dimensão física total da janela em mm (ex: 2.0 mm -> de -1.0 mm a +1.0 mm).
    wavelength_um : float
        Comprimento de onda em micrômetros (ex: 0.6238 um = 623.8 nm).
    w0_um : float
        Raio da cintura (waist) do feixe em micrômetros (ex: 250.0 um).
    periodo_rede_px : float
        Período da rede de difração linear em pixels (0 para desativar).
        Útil para criar máscaras de garfo (forked grating) em SLM.
    pasta_saida : str
        Pasta onde as imagens geradas serão armazenadas.
    salvar_individuais : bool
        Se True, gera PNGs para cada modo (n, l) isoladamente.
    salvar_grids : bool
        Se True, gera painéis combinados com todos os modos.
    """
    os.makedirs(pasta_saida, exist_ok=True)

    # Malha espacial física
    limite = (tamanho_janela_mm / 2.0) * mm
    x0 = np.linspace(-limite, limite, num_pixels)
    y0 = np.linspace(-limite, limite, num_pixels)
    wavelength = wavelength_um * um
    w0 = w0_um * um

    # Malha de coordenadas para rede de difração opcional (em pixels)
    if periodo_rede_px > 0:
        coord_x = np.linspace(-num_pixels // 2, num_pixels // 2 - 1, num_pixels)
        coord_y = np.linspace(-num_pixels // 2, num_pixels // 2 - 1, num_pixels)
        X_px, _ = np.meshgrid(coord_x, coord_y)
        fase_rede = (2.0 * np.pi / periodo_rede_px) * X_px
    else:
        fase_rede = 0.0

    u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)

    total_n = N + 1
    total_k = K + 1

    # Preparação das figuras comparativas (grids)
    if salvar_grids:
        fig_amp, axs_amp = plt.subplots(
            total_n, total_k, figsize=(3.5 * total_k, 3.5 * total_n), squeeze=False
        )
        fig_amp.suptitle("Amplitude / Intensidade (|u|²)", fontsize=16)

        fig_phase, axs_phase = plt.subplots(
            total_n, total_k, figsize=(3.5 * total_k, 3.5 * total_n), squeeze=False
        )
        fig_phase.suptitle("Fase (rad)", fontsize=16)

    print(f"Gerando máscaras para n in [0, {N}] e l in [0, {K}] (resolução: {num_pixels}x{num_pixels})...")

    for n in range(total_n):
        for k in range(total_k):
            # Gera o feixe Laguerre-Gauss
            u.laguerre_beam(
                A=1,
                n=n,
                l=k,
                r0=(0 * um, 0 * um),
                w0=w0,
                z0=0,
                z=0.01 * um,
            )

            campo = u.u
            intensidade = np.abs(campo) ** 2
            fase_rad = np.angle(campo)

            # Fase mapeada para [0, 2*pi)
            fase_wrapped = np.mod(fase_rad, 2 * np.pi)

            # Máscara 8-bits pura para SLM
            mascara_slm = (fase_wrapped / (2 * np.pi) * 255).astype(np.uint8)

            titulo = f"LG(n={n}, l={k})"

            if salvar_individuais:
                # 1. Máscara de fase SLM pura (sem eixos, exata 1:1 para projetar no SLM)
                caminho_slm = os.path.join(
                    pasta_saida, f"mascara_slm_fase_n{n}_l{k}.png"
                )
                Image.fromarray(mascara_slm).save(caminho_slm)

                # Se houver rede de difração, salva também a versão forked grating
                if periodo_rede_px > 0:
                    fase_garfo = np.mod(fase_rad + fase_rede, 2 * np.pi)
                    mascara_garfo = (fase_garfo / (2 * np.pi) * 255).astype(np.uint8)
                    caminho_garfo = os.path.join(
                        pasta_saida, f"mascara_slm_garfo_n{n}_l{k}.png"
                    )
                    Image.fromarray(mascara_garfo).save(caminho_garfo)

                # 2. Imagem de fase (escala de cinza)
                caminho_fase = os.path.join(
                    pasta_saida, f"fase_n{n}_l{k}.png"
                )
                plt.imsave(caminho_fase, fase_rad, cmap="gray", origin="lower")

                # 3. Imagem de intensidade (gist_heat)
                caminho_int = os.path.join(
                    pasta_saida, f"intensidade_n{n}_l{k}.png"
                )
                plt.imsave(caminho_int, intensidade, cmap="gist_heat", origin="lower")

            if salvar_grids:
                # Subplot Amplitude
                ax_a = axs_amp[n, k]
                ax_a.imshow(intensidade, cmap="gist_heat", origin="lower")
                ax_a.set_title(titulo, fontsize=11)
                ax_a.axis("off")

                # Subplot Fase
                ax_p = axs_phase[n, k]
                ax_p.imshow(fase_rad, cmap="gray", origin="lower")
                ax_p.set_title(titulo, fontsize=11)
                ax_p.axis("off")

    if salvar_grids:
        fig_amp.tight_layout()
        caminho_grid_amp = os.path.join(pasta_saida, "grid_amplitude.png")
        fig_amp.savefig(caminho_grid_amp, dpi=200, bbox_inches="tight")
        plt.close(fig_amp)

        fig_phase.tight_layout()
        caminho_grid_fase = os.path.join(pasta_saida, "grid_fase.png")
        fig_phase.savefig(caminho_grid_fase, dpi=200, bbox_inches="tight")
        plt.close(fig_phase)

    print(f"Sucesso! Arquivos salvos em '{pasta_saida}/'.")


def main():
    parser = argparse.ArgumentParser(
        description="Gera máscaras de fase e intensidade para feixes Laguerre-Gauss."
    )
    parser.add_argument(
        "-N", "--N", type=int, default=0, help="Ordem radial máxima n (padrão: 0)"
    )
    parser.add_argument(
        "-K", "--K", type=int, default=3, help="Carga topológica máxima l/k (padrão: 3)"
    )
    parser.add_argument(
        "--pixels", type=int, default=512, help="Resolução da malha (padrão: 512)"
    )
    parser.add_argument(
        "--janela",
        type=float,
        default=2.0,
        help="Largura total da janela em mm (padrão: 2.0)",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=0.6238,
        help="Comprimento de onda em um (padrão: 0.6238)",
    )
    parser.add_argument(
        "--w0",
        type=float,
        default=250.0,
        help="Cintura do feixe w0 em um (padrão: 250.0)",
    )
    parser.add_argument(
        "--rede",
        type=float,
        default=0.0,
        help="Período da rede linear em pixels para garfo/blazed grating (0 desativa, padrão: 0.0)",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default="mascaras_saida",
        help="Diretório de saída (padrão: mascaras_saida)",
    )

    args = parser.parse_args()

    gerar_mascaras(
        N=args.N,
        K=args.K,
        num_pixels=args.pixels,
        tamanho_janela_mm=args.janela,
        wavelength_um=args.wavelength,
        w0_um=args.w0,
        periodo_rede_px=args.rede,
        pasta_saida=args.saida,
        salvar_individuais=True,
        salvar_grids=True,
    )


if __name__ == "__main__":
    main()
