package mnb

import (
	"errors"
	"math"
)

type Classifier struct {
	cc int       // classes count
	fc int       // features count
	fs []float64 // feature statistics by class
	ft []float64 // features totals by class
	ct []float64
	tc float64
}

func New(classes, features int) *Classifier {
	c := &Classifier{
		cc: classes,
		fc: features,
		fs: make([]float64, classes*features),
		ft: make([]float64, classes),
		ct: make([]float64, classes),
	}
	for i := range c.fs {
		c.fs[i] = 1
	}
	for i := range c.ft {
		c.ft[i] = 1
	}
	return c
}

func (c *Classifier) Learn(class int, fv []float64) error {
	if class < 0 || class >= c.cc {
		return errors.New("unknown class")
	}
	c.tc += 1
	c.ct[class] += 1
	base := class * c.fc
	fc := len(fv)
	if fc > c.fc {
		fc = c.fc
	}
	for fi := 0; fi < fc; fi++ {
		c.fs[base+fi] += fv[fi]
		c.ft[class] += fv[fi]
	}
	return nil
}

func (c *Classifier) Predict(fv []float64) ([]float64, error) {
	score := make([]float64, c.cc)
	for ci := 0; ci < c.cc; ci++ { // for each class
		base := ci * c.fc
		cf := c.fs[base : base+c.fc] // slice feature statistics by class
		cv := 0.0                    // calculated class value
		fc := len(fv)
		if fc > c.fc {
			fc = c.fc
		}
		for fi := 0; fi < fc; fi++ { // for each feature
			cv += (math.Log(cf[fi]) - math.Log(c.ft[ci])) * fv[fi]
		}
		score[ci] = cv + math.Log(c.ct[ci]/c.tc)
	}
	return score, nil
}
