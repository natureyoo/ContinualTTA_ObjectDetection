import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter

fig_size = (18,15)
fig6, ax6 = plt.subplots(dpi=150, figsize=fig_size)
markersize=700
linewidth = 6
text_font_size = 45
# color = 'dimgray'
marker_r101 = 'v'


color = 'green'
ours = ([22.8, 17.3, 14.0, 11.7], [46.9, 48.2, 48.1, 49.4])
ax6.plot(ours[0][:2], ours[1][:2], color=color, markersize=markersize, linewidth=linewidth, label='Ours #300')
ax6.scatter(ours[0][0], ours[1][0], color=color, s=markersize)
ax6.scatter(ours[0][1], ours[1][1], color=color, marker=marker_r101, s=markersize)
ax6.text(17.6, 48.3, s='Ours #300', color=color, fontsize=text_font_size)

num_2000_color = 'red'
ax6.plot(ours[0][2:], ours[1][2:], color=num_2000_color, markersize=markersize, linewidth=linewidth, label='Ours #2000 Top-0.7')
ax6.text(12.0, 49.5, s='Ours #2000 Top-0.7', color=num_2000_color, fontsize=text_font_size)
ax6.scatter(ours[0][2], ours[1][2], color=num_2000_color, s=markersize)
ax6.scatter(ours[0][3], ours[1][3], color=num_2000_color, marker=marker_r101, s=markersize)

srcnn_color = 'blue'
srcnn = ([22.7, 17.3], [45.2, 46.4])
ax6.plot(srcnn[0], srcnn[1], color=srcnn_color, markersize=markersize, linewidth=linewidth, label='Sparse R-CNN #300')
ax6.scatter(srcnn[0][0], srcnn[1][0], color=srcnn_color, s=markersize)
ax6.scatter(srcnn[0][1], srcnn[1][1], color=srcnn_color, marker=marker_r101, s=markersize)
# ax6.text(17.70, 46.5, s='Sparse R-CNN', color=srcnn_color, fontsize=text_font_size)

queryinst_color = 'orange'
queryinst = ([22.7, 17.3], [46.9, 48.1])
ax6.plot(queryinst[0], queryinst[1],markersize=markersize,  color=queryinst_color, linewidth=linewidth, label='Query Inst #300')
ax6.scatter(queryinst[0][0], queryinst[1][0], color=queryinst_color, s=markersize)
ax6.scatter(queryinst[0][1], queryinst[1][1], color=queryinst_color, marker=marker_r101, s=markersize)

deform_color = 'cyan'
deformdetr = ([13.5, 11.1], [46.9, 48.7])
ax6.plot(deformdetr[0], deformdetr[1], color=deform_color,markersize=markersize, linewidth=linewidth, label='Deformble DETR #300')
ax6.scatter(deformdetr[0][0], deformdetr[1][0], color=deform_color, s=markersize)
ax6.scatter(deformdetr[0][1], deformdetr[1][1], color=deform_color, marker=marker_r101, s=markersize)


reppointv2_color = 'pink'
reppointv2 = ([16.2, 13.1], [44.4, 46.0])
ax6.plot(reppointv2[0], reppointv2[1], color=reppointv2_color,markersize=markersize, linewidth=linewidth, label='Reppoints V2')
ax6.scatter(reppointv2[0][0], reppointv2[1][0], color=reppointv2_color, s=markersize)
ax6.scatter(reppointv2[0][1], reppointv2[1][1], color=reppointv2_color, marker=marker_r101, s=markersize)


gfocalv2_color = 'purple'
gfocalv2 = ([21.2, 16.3], [44.3, 46.2])
ax6.plot(gfocalv2[0], gfocalv2[1], color=gfocalv2_color,markersize=markersize, linewidth=linewidth, label='GFocalV2')
ax6.scatter(gfocalv2[0][0], gfocalv2[1][0], color=gfocalv2_color, s=markersize)
ax6.scatter(gfocalv2[0][1], gfocalv2[1][1], color=gfocalv2_color, marker=marker_r101, s=markersize)


ax6.scatter([0],[0], color='black', s=250, label='ResNet50 backbone')
ax6.scatter([0],[0], color='black', marker=marker_r101, s=250, label='ResNet101 backbone')

label_font_size = 50
tick_font_size = 40


ax6.set_xlabel('FPS (TITAN RTX)', fontsize=label_font_size)
ax6.set_ylabel('AP', fontsize=label_font_size)
ax6.set_ylim(44, 50.0)
ax6.set_xlim(10, 24.0)
ax6.grid(color='gray', linestyle='-')



plt.xticks(fontsize=tick_font_size, rotation=0)
plt.yticks(fontsize=tick_font_size, rotation=0)
y_label = [44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]
ax6.set_yticks(y_label)
ax6.set_yticklabels(y_label, fontsize=tick_font_size)
#
# x_label = [10.0, 12.5, 15.0, 17.5, 20.0, 22.5]
# x_label = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
# ax6.set_xticks(x_label)
# ax6.set_xticklabels(x_label, fontsize=tick_font_size)


ax6.legend(fontsize=45.0, fancybox=True, framealpha=1.0,  bbox_to_anchor=(1.0, 1.025))

# plt.show()

# fig6.patch.set_alpha(0.0)
fig6.savefig("graph_comparison.pdf", dpi=150, bbox_inches='tight')
fig6.savefig("graph_comparison.png", dpi=150, bbox_inches='tight', facecolor='white', transparent=False)