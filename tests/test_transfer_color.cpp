#include "../utils.h"

int main(int argc, char** argv) {
  QImage source(argv[1]);
  QImage target(argv[2]);

#if 0
  vector<int> pix_s(source.width()*source.height()), pix_t(target.width()*target.height());
  for(int i=0;i<source.width()*source.height();++i) pix_s[i] = i;
  for(int i=0;i<target.width()*target.height();++i) pix_t[i] = i;
#else
  vector<int> pix_s, pix_t;
  for(int i=0;i<target.width()*target.height();++i) {
    QRgb pix = target.pixel(i%target.width(), i/target.width());
    if(qAlpha(pix) > 0) {
      pix_t.push_back(i);
    }
  }
  pix_s = pix_t;
#endif

  QImage transferred = TransferColor(source, target, pix_s, pix_t);

  source.save("source.png");
  target.save("target.png");
  transferred.save("transferred.png");

  return 0;
}
